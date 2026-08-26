"""#871c: which pinned-host entry point ADMITS, and which only COUNTS.

`pinned_host_budget` has two entry points and they differ in the one way that
matters:

    register_pinned_post(post)              -> counts. Never refuses.
    check_and_register_pinned_post(...)     -> ADMITS, or raises naming posts.

A comment inside `check_and_register_pinned_post` asserted that the largest
claimant in the system -- the phase-flip host weight images, ~27 GiB on this
box -- is refused "at the DECLARATION rather than discovering it at the
allocation". It is not. `weights_arena` calls the COUNTING entry point, and
never the admitting one (`grep -c check_and_register_pinned_post
weights_arena.py` -> 0). Its own `_register_image_post` docstring says "Never
raises."

THE BEHAVIOUR IS CORRECT AND DELIBERATE; only the comment was wrong. #695 chose
it at the call site: "Registered, not CHECKED: a new refusal path here could
break a boot that works today, and the diagnosis this is for is served by the
number being present, not by a veto." That decision stands and is not touched.

WHY THIS IS TESTED AND NOT JUST RE-COMMENTED. A comment promising a veto that
does not exist is worse than no comment: someone sizing a new pinned pool
against this module would believe the biggest post is admitted when it is only
counted, and would leave exactly the headroom that post already spent. Prose
cannot be regression-tested; the CONTRACT can. So this pins the split itself --
counting never refuses, admitting does -- in both directions, so a future change
that quietly turns one into the other fails here rather than in a boot.

This is the #464/#578/#852 shape stated exactly: a counter and an actuator are
different things, and the danger is a counter DESCRIBED as an actuator.

Hermetic: pure registry operations. No CUDA, no pools, no boot.
"""

import inspect
import unittest

from sglang.srt.mem_cache import pinned_host_budget as B
from sglang.srt.mem_cache.pinned_host_budget import (
    PinnedHostPost,
    check_and_register_pinned_post,
    clear_registered_posts,
    register_pinned_post,
    registered_posts,
)
from sglang.test.test_utils import CustomTestCase

GIB = 1024**3


class TestPinnedPostAdmissionContract(CustomTestCase):
    def setUp(self):
        clear_registered_posts()
        self.addCleanup(clear_registered_posts)

    # -- the split itself ---------------------------------------------------

    def test_counting_never_refuses_however_absurd_the_post(self):
        """`register_pinned_post` is a counter. It must not grow a veto.

        A terabyte on a 118 GiB box is admitted without complaint, because this
        entry point does not admit at all -- it records.
        """
        register_pinned_post(
            PinnedHostPost(name="absurd", flag="--x", nbytes=1024 * GIB)
        )
        self.assertEqual([p.name for p in registered_posts()], ["absurd"])

    def test_admitting_DOES_refuse_and_names_every_post(self):
        """The other half, and the one an operator's boot depends on."""
        with self.assertRaises(ValueError) as ctx:
            check_and_register_pinned_post(
                name="way too big",
                flag="--y",
                requested_bytes=4096 * GIB,
                reserve_bytes=10 * GIB,
            )
        self.assertIn("way too big", str(ctx.exception))
        self.assertIn("--y", str(ctx.exception))

    def test_a_refused_post_is_NOT_left_in_the_registry(self):
        """Otherwise the next admission is charged for bytes never allocated."""
        try:
            check_and_register_pinned_post(
                name="refused one", flag="--z", requested_bytes=4096 * GIB
            )
        except ValueError:
            pass
        self.assertNotIn("refused one", [p.name for p in registered_posts()])

    # -- the producer the false comment named --------------------------------

    def test_weights_arena_uses_the_COUNTING_entry_point_only(self):
        """The fact the corrected comment now states. Pinned as behaviour.

        If someone later wires the admitting entry point in there, that is a
        real decision with a boot-refusal consequence and it must not slip in
        unnoticed -- this test is where it surfaces.
        """
        from sglang.srt.model_executor import weights_arena as WA

        src = inspect.getsource(WA)
        self.assertIn("register_pinned_post", src)
        self.assertNotIn(
            "check_and_register_pinned_post",
            src,
            "weights_arena now ADMITS its image post. That is a behaviour "
            "change #695 explicitly declined ('a new refusal path here could "
            "break a boot that works today'); if it is intended, update this "
            "test and the comment in pinned_host_budget that describes it.",
        )

    def test_the_image_post_declares_that_it_never_raises(self):
        from sglang.srt.model_executor import weights_arena as WA

        self.assertIn("Never raises", WA._register_image_post.__doc__ or "")

    # -- the comment that promised a veto ------------------------------------

    def test_the_correction_is_recorded_where_the_false_claim_was(self):
        """POSITIVE ONLY, and the reason is a rule I proved on myself.

        The first version of this test asserted the false sentence was ABSENT
        from the module. It failed against the FIXED file -- because a
        correction has to quote the claim it corrects, so the corrected text
        contains the wrong sentence by necessity. That is the third instance in
        one session of the same mistake: an assertion blunt enough to catch the
        EXPLANATION is worthless the moment someone documents the fix (#871a's
        floor promise, #871b's /proc/meminfo probe, and now this).

        The general rule it proves: ABSENCE OF PROSE IS NOT TESTABLE where a
        correction must restate what it corrects. So prose gets one positive
        marker, and the real contract is pinned by the BEHAVIOUR tests above --
        which is what those tests are for.
        """
        src = inspect.getsource(B)
        self.assertIn(
            "THE REGISTRY REFUSES NOTHING THERE",
            src,
            "the correction naming `register_pinned_post` as a counter rather "
            "than a veto is gone from the module",
        )


if __name__ == "__main__":
    unittest.main()
