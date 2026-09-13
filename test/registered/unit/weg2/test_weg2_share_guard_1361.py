# SPDX-License-Identifier: Apache-2.0
"""#1361 [23c] -- W89x was named, documented and unreachable.

`_derive_d_l2_budget` read:

    share = float(ns.d_cap_rank_share)
    if rec_share is not None:
        share = rec_share                        # the overwrite
    if rec_share is not None and rec_share > share + 1e-9:
        raise Weg2L2ShareBelowInstalled(...)     # compares y against y

`share` had just been set to `rec_share`, so the condition could not be true on
any input. The refusal existed in the source, in the W-code census and in the
comments, and could never fire.

That is worse than no refusal. Every reader downstream -- this seat included,
twice in one day -- takes a named guard as proof the case is covered. It is the
same family as a reader nobody calls and a comment that promises a property the
code does not have, except that here the condition is killed by its own
precedent rather than by a missing consumer.

The comparison now happens BEFORE the overwrite, against the FLAG value, which
is what the refusal text always claimed it compared against ("above
--d-cap-rank-share {share}").
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as lc
from sglang.test.test_utils import CustomTestCase


def _ns(flag_share):
    return SimpleNamespace(d_cap_rank_share=flag_share)


class TheShareGuardCanActuallyFire(CustomTestCase):
    def test_a_record_share_above_the_flag_is_refused_by_name(self):
        """THE CASE THAT COULD NEVER FIRE. Red on the dead form."""
        with mock.patch.object(lc, "_installed_max_share_from_record",
                               return_value=0.5000):
            with self.assertRaises(lc.Weg2LaunchRefused) as cm:
                lc._derive_d_l2_budget(_ns(0.4000), 262144)
        msg = str(cm.exception)
        self.assertIn("W89x", msg)
        self.assertIn("0.5000", msg)
        self.assertIn("0.4", msg, "the refusal must name the FLAG it compared against")
        # BOTH NUMBERS AND BOTH SOURCES. The record overrides the flag a few
        # lines below this guard, so a refusal naming one value leaves the
        # reader unable to tell which one sized the arm -- exactly the gap that
        # let the boot seat read S_D=1 while the ARM line said S_D=2.
        self.assertIn("source=record", msg)
        self.assertIn("source=flag", msg)

    def test_a_record_share_at_or_below_the_flag_still_passes(self):
        """The control: the guard must not refuse the ordinary case.

        weg2sn6p measured installed 0.3750 against the shipped flag default
        0.4 -- the configuration every boot on this rig has run.
        """
        with mock.patch.object(lc, "_installed_max_share_from_record",
                               return_value=0.3750):
            s_gb_d, terms, _ = lc._derive_d_l2_budget(_ns(0.4000), 262144)
        self.assertGreater(int(s_gb_d), 0)
        self.assertEqual(terms["max_rank_share_source"], "record")
        self.assertAlmostEqual(float(terms["max_rank_share"]), 0.3750, places=6)

    def test_without_a_record_the_flag_stands_and_says_so(self):
        with mock.patch.object(lc, "_installed_max_share_from_record",
                               return_value=None):
            _, terms, _ = lc._derive_d_l2_budget(_ns(0.4000), 262144)
        self.assertEqual(terms["max_rank_share_source"], "flag")
        self.assertAlmostEqual(float(terms["max_rank_share"]), 0.4000, places=6)

    def test_the_guard_is_not_compared_against_its_own_result(self):
        """THE SHAPE, pinned in source.

        A behavioural test alone would go green again the day someone re-orders
        these lines, because the refusal would simply stop firing on inputs no
        test happens to supply. The order is the property.
        """
        import inspect

        # CODE LINES ONLY. The first draft of this assertion searched the raw
        # source and matched `share = rec_share` inside the comment that
        # EXPLAINS the defect -- the same prose trap as #1362 [22-fix3], where
        # a refusal quoted the marker of the line it was describing. An
        # instrument that cannot tell its subject from a description of its
        # subject reports on the description.
        src = "\n".join(
            ln for ln in inspect.getsource(lc._derive_d_l2_budget).splitlines()
            if not ln.strip().startswith("#")
        )
        guard = src.index("W89x")
        overwrite = src.index("share = rec_share")
        self.assertLess(
            guard, overwrite,
            "the W89x comparison must come BEFORE `share = rec_share`; "
            "after it, the condition compares rec_share against itself",
        )


if __name__ == "__main__":
    unittest.main()
