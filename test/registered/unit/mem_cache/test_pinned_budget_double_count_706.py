# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The runtime pinned-host backstop must not bill allocated posts twice.

THE SPECIMEN, 2026-08-17. The Flip+HiCache run-card boot (DESIGN_706_BOOT §5)
refused on PP0:

    Pinned host RAM over-committed: 40.42 GB requested across 4 pool(s)
    [phase-flip host weight image #1 18.06 GB; #2 17.12 GB; ...;
     MHATokenToKVPoolHost ...] does not fit in 33.97 GB available minus a
    10.74 GB OS reserve = 23.23 GB usable.

35.18 GB of that demand was the weight images -- which #695 registers AFTER
allocating them (weights_arena.py:428). So those bytes were already gone from
`available` when it was read, and were then added to the demand as well.

Measured against the live serving process the same day: MemAvailable 33.62 GB
while the three schedulers held 86.51 GB resident on a 126.75 GB box. The
images were in RSS. The boot's true marginal cost was the tier alone.

WHAT IS NOT CHANGED: `joint_pinned_host_error` itself. It is exact where it was
designed to be used -- once in the launcher, over configured numbers, before
anything is pinned. Only the RUNTIME backstop, which runs after allocations,
needs the credit.
"""

import unittest
from unittest import mock

from sglang.srt.mem_cache import pinned_host_budget
from sglang.srt.mem_cache.pinned_host_budget import (
    PINNED_HOST_RESERVE_BYTES,
    PinnedHostPost,
    check_and_register_pinned_post,
    clear_registered_posts,
    joint_pinned_host_error,
    register_pinned_post,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

GB = 10**9
#: The PP0 refusal, to the same precision the log printed it.
IMAGES_GB = (18.06, 17.12)
TIER_GB = 5.24
AVAILABLE_GB = 33.97
TOTAL_GB = 126.75


class TheSpecimenNoLongerRefuses(unittest.TestCase):
    def setUp(self):
        clear_registered_posts()
        self.addCleanup(clear_registered_posts)
        # PIN THE MACHINE OUT OF THE TEST. Read live, `available` on a roomy
        # box would admit the tier even WITHOUT the fix, and the can-fail
        # below would prove nothing there while still passing here. These are
        # the specimen's own numbers, so the discrimination is the same
        # everywhere the suite runs.
        patcher = mock.patch.object(
            pinned_host_budget,
            "pinned_host_memory_bytes",
            return_value=(int(TOTAL_GB * GB), int(AVAILABLE_GB * GB)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _register_images(self):
        """The images, as #695 registers them: AFTER they are allocated."""
        for i, gb in enumerate(IMAGES_GB, start=1):
            register_pinned_post(
                PinnedHostPost(
                    name=f"phase-flip host weight image #{i}",
                    flag="--enable-phase-flip",
                    nbytes=int(gb * GB),
                )
            )

    def test_the_old_comparison_refuses_the_specimen(self):
        """RED-FIRST: the arithmetic that blocked the boot, pinned as such.

        This calls `joint_pinned_host_error` the way the backstop used to --
        every post against the post-allocation availability -- and asserts it
        refuses. If someone 'fixes' the shared helper instead of the caller,
        this goes green and says so.
        """
        posts = [
            PinnedHostPost(f"image #{i}", "--enable-phase-flip", int(gb * GB))
            for i, gb in enumerate(IMAGES_GB, start=1)
        ] + [PinnedHostPost("MHATokenToKVPoolHost", "--hicache-size", int(TIER_GB * GB))]
        err = joint_pinned_host_error(
            posts,
            int(TOTAL_GB * GB),
            int(AVAILABLE_GB * GB),
            PINNED_HOST_RESERVE_BYTES,
        )
        self.assertIsNotNone(err, "the specimen must refuse under the old rule")
        self.assertIn("over-committed", err)

    def test_the_backstop_admits_the_tier_now(self):
        """The fix: with the images already allocated, only the tier is new."""
        self._register_images()
        check_and_register_pinned_post(
            name="MHATokenToKVPoolHost",
            flag="--hicache-size / --hicache-ratio",
            requested_bytes=int(TIER_GB * GB),
        )  # must not raise

    def test_a_genuinely_impossible_tier_is_still_refused(self):
        """CAN-FAIL: crediting must not disable the guard.

        A tier larger than the real free memory has to fail, or this change
        would have replaced a false refusal with no protection at all -- which
        is the worse defect, since pinned memory is non-swappable on this box.
        """
        self._register_images()
        with self.assertRaises(ValueError) as cm:
            check_and_register_pinned_post(
                name="MHATokenToKVPoolHost",
                flag="--hicache-size / --hicache-ratio",
                requested_bytes=400 * GB,
            )
        self.assertIn("over-committed", str(cm.exception))

    def test_the_refusal_still_prices_every_post(self):
        """The module's rule: an operator must see which flag to lower."""
        self._register_images()
        with self.assertRaises(ValueError) as cm:
            check_and_register_pinned_post(
                name="MHATokenToKVPoolHost",
                flag="--hicache-size / --hicache-ratio",
                requested_bytes=400 * GB,
            )
        msg = str(cm.exception)
        self.assertIn("phase-flip host weight image #1", msg)
        self.assertIn("--enable-phase-flip", msg)
        self.assertIn("--hicache-size", msg)

    def test_the_post_being_weighed_is_not_credited_to_itself(self):
        """The new post is not yet allocated, so it must stay pure demand.

        Crediting it too would make the guard admit anything.
        """
        clear_registered_posts()
        with self.assertRaises(ValueError):
            check_and_register_pinned_post(
                name="solo",
                flag="--hicache-size",
                requested_bytes=10_000 * GB,
            )

    def test_first_post_in_a_process_is_unaffected(self):
        """With nothing registered, credit is zero and behaviour is identical."""
        clear_registered_posts()
        check_and_register_pinned_post(
            name="first", flag="--hicache-size", requested_bytes=1
        )


if __name__ == "__main__":
    unittest.main()
