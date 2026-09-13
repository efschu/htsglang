# SPDX-License-Identifier: Apache-2.0
"""#1358 -- the bounce depth is HOST STAGING and must not move the P form key.

The train seat measured that `--xchg-bounce-depth` re-spelled the key
(without: fbcf124aabb5, with: 7824e006efd8). Left there, xsn28 would meet W48
against weg2xsn27's ring table and re-solve a census that is already correct
for it.

WHY IT IS EXCLUDED, and the reason has to be the criterion rather than the
inconvenience: the flag sizes the assemble buffer in /dev/shm
(`widest_layer x assemble_slots(depth)`). It says nothing about WHICH weights
group P loads or how they are cut, and Sigma H is the DORMANT IMAGE -- the
backed-up tag census -- which does not contain the staging buffer at all. A
ring table solved under depth=2 is a measurement OF THIS FORM under depth=1.

THE CONTRAST IS PINNED IN THE SAME FILE ON PURPOSE. #1356's `--weg2-vision off`
STAYS IN the key: an absent tower changes the tag census by 2.63 GiB, so it IS
a different weight statement and its first boot must solve from its own stem
(W48 expected once, not W20). "Exclude the flags that are noisy" is how a form
key stops discriminating; the question is never whether a flag is ours, it is
whether it moves Sigma H.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ring_table as rt
from sglang.test.test_utils import CustomTestCase

BASE = ["py", "-m", "sglang.launch_server", "--model-path", "/m",
        "--tp-size", "1", "--pp-size", "3", "--port", "30031"]


class TheBounceDepthDoesNotMoveTheKey(CustomTestCase):
    def test_the_key_is_identical_with_and_without_the_flag(self):
        k0, _ = rt.p_form_key(BASE)
        for extra in (["--xchg-bounce-depth", "1"],
                      ["--xchg-bounce-depth", "2"],
                      ["--xchg-bounce-depth=1"]):
            with self.subTest(spelling=extra):
                k, f = rt.p_form_key(BASE + extra)
                self.assertEqual(k, k0, "the bounce depth re-spelled the form key")
                self.assertNotIn("xchg-bounce-depth", f)

    def test_it_is_in_the_exclusion_list(self):
        self.assertIn("--xchg-bounce-depth", rt.FORM_KEY_EXCLUDED_FLAGS)


class TheVisionFlagStaysInTheKey(CustomTestCase):
    """The other half of the criterion -- an absent tower IS a weight change."""

    def test_vision_off_is_NOT_excluded(self):
        for flag in ("--weg2-vision", "--no-enable-multimodal",
                     "--enable-multimodal"):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, rt.FORM_KEY_EXCLUDED_FLAGS)

    def test_a_text_only_argv_hashes_differently(self):
        k0, _ = rt.p_form_key(BASE)
        k1, _ = rt.p_form_key(BASE + ["--no-enable-multimodal"])
        self.assertNotEqual(
            k0, k1,
            "a tower that is absent must produce a different form key -- its "
            "first boot has to solve the ring from its own stem")

    def test_the_two_rules_are_independent(self):
        """Depth excluded AND vision included, on one argv."""
        k_a, _ = rt.p_form_key(BASE + ["--no-enable-multimodal"])
        k_b, _ = rt.p_form_key(
            BASE + ["--no-enable-multimodal", "--xchg-bounce-depth", "1"])
        self.assertEqual(k_a, k_b)


if __name__ == "__main__":
    unittest.main()
