"""#239 S4b (F14) x #706: the canonical KV page under the Form A token cut.

rc12z29 (f5b672124f, profile -st-cut, 28.09. 19:08Z) died in group D's
argument parsing before a single card was touched:

    ValueError: --hicache-canonical-kv-page requires --page-size 1 under
    weighted uneven DCP, got 64.

The refusal predates the token cut. Under the cut a 64-token page DOES span
owners (global slot L belongs to the rank with L % S in [lo, hi); the D argv
``--uneven-token-vector 0,48,16`` resolves to [0, 3, 1], S = 4), but since S4b
(F14) a page is no longer one owner's: every owner writes its own token rows of
each page (``canonical_page_store.owner_token_runs`` / ``kv_extents_for``), and
the runtime twin in ``HiCacheController`` already lifts the page-1 limit when
``canonical_kv_owner_rows`` is set. What must still hold:

(1) the D argv of rc12z29 -st-cut (canonical page + page 64 + token cut)
    passes argument parsing;
(2) a token cut whose S does not divide the page is refused by name -- the
    owned rows would differ from page to page;
(3) weighted uneven DCP WITHOUT the token cut keeps the page-1 refusal.
"""

import argparse
import dataclasses
import os
import unittest
from unittest import mock

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# the flags of the dead D launch that decide the clause (D.log line 2)
RC12Z29_D_ARGV = [
    "--model-path", "dummy",
    "--tp-size", "3",
    "--page-size", "64",
    "--hicache-storage-backend", "file",
    "--hicache-canonical-kv-page",
    "--rank-role", "host,worker,worker",
    "--rank-tp-ratio", "1,0,0",
    "--uneven-token-vector", "0,48,16",
]


def _parse(argv):
    """argv -> ServerArgs the way launch_server builds it. ``model_path='dummy'``
    short-circuits ``__post_init__``; the two handlers the real boot runs in
    this order are called explicitly."""
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(argv)
    fields = {f.name for f in dataclasses.fields(ServerArgs)}
    return ServerArgs(**{k: v for k, v in vars(ns).items() if k in fields})


def _with(argv, flag, value):
    out = list(argv)
    out[out.index(flag) + 1] = value
    return out


class TestCanonicalPageUnderTokenCut(CustomTestCase):
    def test_rc12z29_d_argv_is_accepted(self):
        args = _parse(RC12Z29_D_ARGV)
        args._resolve_form_a_dcp()
        self.assertEqual(args.form_a_dcp_vector(), [0, 3, 1])
        self.assertTrue(args.uneven_weighted_dcp_enabled())
        args._handle_hicache_canonical_kv_page()  # no raise
        self.assertTrue(args.hicache_canonical_kv_page)

    def test_a_split_that_does_not_divide_the_page_is_refused_by_name(self):
        args = _parse(_with(RC12Z29_D_ARGV, "--page-size", "2"))
        args._resolve_form_a_dcp()
        with self.assertRaisesRegex(ValueError, r"token cut \[0, 3, 1\].*divisible by S=4"):
            args._handle_hicache_canonical_kv_page()

    def test_weighted_uneven_dcp_without_the_cut_keeps_page_one(self):
        argv = [
            "--model-path", "dummy", "--tp-size", "3", "--page-size", "64",
            "--hicache-storage-backend", "file", "--hicache-canonical-kv-page",
        ]
        with mock.patch.dict(os.environ, {"SGLANG_UNEVEN_DCP_WEIGHTED": "1"}):
            args = _parse(argv)
            self.assertIsNone(args.form_a_dcp_vector())
            with self.assertRaisesRegex(ValueError, "requires --page-size 1 under weighted uneven DCP"):
                args._handle_hicache_canonical_kv_page()


if __name__ == "__main__":
    unittest.main()
