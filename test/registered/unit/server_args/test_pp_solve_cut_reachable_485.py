"""The --pp-solve-cut path must be REACHABLE, not merely written.

#485. `_handle_pp_solve_cut` calls `self._pp_cut_token_shares()`, and no such
method exists on ServerArgs. Every boot passing `--pp-solve-cut` therefore died
with

    AttributeError: 'ServerArgs' object has no attribute '_pp_cut_token_shares'

before the gate it exists to feed ever ran.

WHY IT SURVIVED. The only harness that ever exercised the gate --
`/spinning/evidence-631/s51/gate_from_census.py` -- rebuilds `RankResources` by
hand "the way `_handle_pp_solve_cut` builds them". A mirror of a code path
cannot discover that the path does not run: it reproduces the arithmetic and
skips the dispatch. The desk validation in RUNSHEET 485 section 9 is all
`certify_485.py` and `excursion_485.py`, neither of which constructs a
ServerArgs. So the flag was verified to PARSE (section 4, "8/8 PASS ... by
building the real parser") and never verified to RUN.

So the first test here is deliberately not about token shares. It asserts the
whole class: every `self._pp_cut_*` name the solve body calls must exist. A
second missing helper would be the same outage with a different traceback, and
naming them one at a time is how a corpus accumulates seven of these.
"""

import inspect
import os
import re
import unittest

from sglang.srt.planner import pp_cut
from sglang.srt.server_args import ServerArgs

_SELF_CALL = re.compile(r"self\.(_pp_cut_[A-Za-z0-9_]+)\s*\(")


class ThePPSolveCutPathIsReachableTest(unittest.TestCase):
    def test_every_pp_cut_helper_the_solve_body_calls_exists(self):
        src = inspect.getsource(ServerArgs._handle_pp_solve_cut)
        called = sorted(set(_SELF_CALL.findall(src)))
        self.assertTrue(
            called, "the solve body calls no _pp_cut_* helper -- regex is stale"
        )
        missing = [n for n in called if not hasattr(ServerArgs, n)]
        self.assertEqual(
            [],
            missing,
            f"--pp-solve-cut calls {missing} and ServerArgs defines no such "
            f"attribute, so the flag raises AttributeError before the cut gate "
            f"runs. Helpers the body calls: {called}",
        )

    def test_token_shares_come_from_the_resolved_token_vector(self):
        """Not the flip WEIGHT vector -- token_shares_from_vector says so."""
        sa = ServerArgs.__new__(ServerArgs)
        sa.pp_size = 3
        sa.phase_flip_tp_vector = "32,16,16"
        prev = os.environ.get("SGLANG_UNEVEN_TOKEN_VECTOR")
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "10,3,3"
        try:
            shares = sa._pp_cut_token_shares()
        finally:
            if prev is None:
                os.environ.pop("SGLANG_UNEVEN_TOKEN_VECTOR", None)
            else:
                os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = prev
        self.assertEqual(len(shares), 3)
        self.assertAlmostEqual(sum(shares), 1.0, places=9)
        self.assertEqual(shares, pp_cut.token_shares_from_vector((10, 3, 3)))
        # and NOT the weight vector, which would normalise to a different split
        self.assertNotEqual(shares, pp_cut.token_shares_from_vector((32, 16, 16)))

    def test_a_gcd_reduced_vector_is_the_same_vector(self):
        """14,10,8 and 7,5,4 are one configuration, per the helper's contract."""
        self.assertEqual(
            pp_cut.token_shares_from_vector((14, 10, 8)),
            pp_cut.token_shares_from_vector((7, 5, 4)),
        )


if __name__ == "__main__":
    unittest.main()
