# SPDX-License-Identifier: Apache-2.0
"""#1305 item 4 (boot weg2sn5pre): the ring's SOURCE sees the xchg arm, and
other-form candidates rank by form DISTANCE, not by age.

Boots weg2she1 (an xchg shadow boot) and weg2sb5d (a serving boot) hashed to
the SAME group-P form key -- the weight-exchange region is a launcher-level shm
region plus hooks, not a P flag -- while their measured dormant P images
differed by 5171 MiB.  A defaults boot at the serve tip read its ring from
whichever was newer by mtime: she1, and refused W20 at the launch moment
(-1.31 GiB); pinned to sb5d it funded (+3.57 GiB).  Two changes, both in
``ring_table``: (1) ``parse_p_form`` folds the xchg launcher's own marker into
the source's form as a synthetic flag, so the key discriminates; (2) among
other-form candidates the ranking is by the number of differing form terms,
then age, and an xchg source is excluded outright for a boot that does not
carry the token.  Hermetic: temp files only.
"""

import inspect
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ring_table
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ARGV_LINE = (
    "[2026-09-09T15:11:45Z] WEG2-LAUNCH group P argv: /venv/bin/python -m "
    "sglang.launch_server --model-path /m --tp-size 1 --pp-size 3 "
    "--pp-stage-ratio 32,18,14 --pp-attn-stage-ratio 8,4,4 --port 30031\n"
)
XCHG_LINE = (
    "[2026-09-09T15:11:49Z] WEG2-LAUNCH WEG2-XCHG-REGION epoch=1788966709 "
    "path=/dev/shm/weg2-xchg-1788966709/xchg.bin slots=6x2x32MiB\n"
)


def _front(*lines):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "boot_weg2_x_0000000000_0909_000000.front.log")
    with open(p, "w") as fh:
        fh.writelines(lines)
    return p


class TheXchgArmIsAFormTerm(CustomTestCase):
    def test_a_serving_source_carries_no_token(self):
        argv, why = ring_table.parse_p_form(_front(ARGV_LINE))
        self.assertEqual(why, "")
        self.assertNotIn(ring_table.XCHG_FORM_TOKEN, argv)

    def test_an_xchg_source_carries_the_token_wherever_the_marker_sits(self):
        for lines in ((ARGV_LINE, XCHG_LINE), (XCHG_LINE, ARGV_LINE)):
            argv, why = ring_table.parse_p_form(_front(*lines))
            self.assertEqual(why, "")
            self.assertEqual(argv[-1], ring_table.XCHG_FORM_TOKEN)

    def test_the_token_moves_the_key(self):
        """she1 and sb5d had ONE key; with the token they have two."""
        serving, _ = ring_table.parse_p_form(_front(ARGV_LINE))
        xchg, _ = ring_table.parse_p_form(_front(ARGV_LINE, XCHG_LINE))
        self.assertNotEqual(ring_table.p_form_key(serving)[0], ring_table.p_form_key(xchg)[0])
        self.assertEqual(
            set(ring_table.p_form_key(xchg)[1].split(" ")) - set(ring_table.p_form_key(serving)[1].split(" ")),
            {ring_table.XCHG_FORM_TOKEN},
        )

    def test_the_token_is_not_on_the_exclusion_list(self):
        self.assertNotIn(ring_table.XCHG_FORM_TOKEN.split("=")[0], ring_table.FORM_KEY_EXCLUDED_FLAGS)


class TheRankingIsWired(CustomTestCase):
    def test_same_form_first_then_distance_then_age_and_xchg_excluded(self):
        src = inspect.getsource(ring_table.solve)
        self.assertIn("same_form + other_form", src)
        self.assertIn("len(mine_toks ^ theirs)", src)
        self.assertIn("scored.sort(key=", src)
        self.assertIn("XCHG_FORM_TOKEN in theirs and not i_am_xchg", src)
        self.assertIn("xchg-shadow source(s) EXCLUDED", src)
        # the excluded sources are named on the SKIPPED lines, never silent
        self.assertIn("EXCLUDED -- an xchg shadow boot", src)


if __name__ == "__main__":
    unittest.main()
