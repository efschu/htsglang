"""#871a: the phase-flip staging pin was an UNPRICED post on pinned host RAM.

``mem_cache/pinned_host_budget.py`` is, in its own first line, "single owner of
the question 'may this PINNED host buffer be allocated?'". It exists because of
#547: *"HiCache sized its host pool from --hicache-ratio/--hicache-size and
kv-session-offload sized its own from --kv-session-offload-host-ram-gib, each
validating against the whole machine as though it were the only claimant. Two
independently plausible budgets can be jointly impossible, and because both
pools are PINNED the over-commit is not a swap -- it is the OOM killer picking
a victim that need not even be this process."*

The #871 phase-flip staging pin -- the newest pinned pool, and after #871 a
TWO-pool one (KV + MAMBA) -- went through none of it. ``phase_flip_boot`` had
zero references to that module. So the pin was exactly the claimant #547
describes: sized alone, summed by nobody. #721 is that outcome observed on this
box (cgroup ``oom_kill=17``, system.slice peak 111.3 of 118 GiB).

WHAT WAS THERE INSTEAD was a ledger LINE, printed after the allocation, saying
``host free after = N GB against the 16 GB floor``. Two things were wrong with
it and both are the same mistake:

* it ENFORCED NOTHING. ``free_g`` was read, formatted, and never compared. The
  words "the POST shrinks if it does not fit -- the FLOOR never does" described
  behaviour no code implemented. An unenforced floor in a ledger line is how
  #721's OOM looked survivable on paper.
* the number DID NOT DENOTE ANYTHING. It shelled out to ``free -g``, and
  ``pinned_host_budget``'s docstring is explicit about why that is void here:
  *"Inside this LXC container that file is synthesised by lxcfs: MemAvailable
  can exceed MemTotal (observed on this rig), and with memory.max unlimited it
  reports the HOST's figures on a box other containers are also spending."*
  ``pinned_host_memory_bytes`` is the #407 declared owner and reads
  ``/sys/fs/cgroup``.

WHAT THIS SUITE PINS

* the KV half is ADMITTED THROUGH THE OWNER BEFORE IT IS PINNED, so an
  over-commitment is a named refusal instead of a later OOM kill;
* a refusal is NOT swallowed -- it becomes the "#847 could not allocate"
  path, which leaves the rebind refusing at the first cutover with a legible
  reason (the guard working as designed) rather than half a tier;
* the post is TAKEN BACK when the allocation fails afterwards. The module
  names this hazard itself: *"a post that is registered and then NEVER
  allocates ... would be credited back without ever having been resident, and
  the next admission would be charged too little -- the registry waving
  through the over-commitment it exists to refuse."*
* the module is reached AT ALL from ``phase_flip_boot`` -- the coarse check
  that would have caught the original bypass.

Hermetic: no CUDA, no pools, no boot. The budget module is driven directly and
``phase_flip_boot`` is read as source, because building a real staging pin needs
a model runner and a device pool.
"""

import inspect
import unittest

from sglang.srt.managers import phase_flip_boot
from sglang.srt.mem_cache import pinned_host_budget
from sglang.srt.mem_cache.pinned_host_budget import (
    PinnedHostPost,
    check_and_register_pinned_post,
    clear_registered_posts,
    joint_pinned_host_error,
    registered_posts,
)
from sglang.test.test_utils import CustomTestCase

GIB = 1024**3


class TestPhasePinIsAPricedPost(CustomTestCase):
    def setUp(self):
        clear_registered_posts()
        self.addCleanup(clear_registered_posts)

    # -- the bypass itself, caught coarsely ---------------------------------

    def test_phase_flip_boot_reaches_the_single_owner(self):
        """The check that would have caught the original defect at desk time.

        Deliberately coarse: it asserts only that the pin's builder consults
        the module that owns pinned-host admission. Before this ticket the
        answer was zero references, and no finer assertion was needed to see
        that -- a pinned pool allocated with no path to the owner is the whole
        bug.
        """
        src = inspect.getsource(phase_flip_boot)
        self.assertIn(
            "pinned_host_budget",
            src,
            "phase_flip_boot allocates PINNED host memory without ever "
            "reaching pinned_host_budget, the declared single owner of that "
            "question. That is #547's defect: a claimant that sizes itself "
            "alone and is summed by nobody.",
        )
        self.assertIn(
            "check_and_register_pinned_post",
            src,
            "the pin is not ADMITTED through the owner, only mentioned near it",
        )

    def test_the_ledger_no_longer_reads_free_dash_g(self):
        """The number in the ledger must be the one the guard decided on.

        `free -g` is void in this container (lxcfs). A ledger and a guard that
        disagree about how much RAM there is are worse than either alone.
        """
        src = inspect.getsource(phase_flip_boot)
        self.assertNotIn(
            '"free", "-g"',
            src,
            "the host ledger still shells out to `free -g`, whose figures are "
            "synthesised by lxcfs and can report the HOST's memory",
        )
        self.assertIn("pinned_host_memory_bytes", src)

    def test_the_ledger_does_not_promise_a_floor_nothing_enforces(self):
        """No unenforced number in the ledger line.

        The old text named a 16 GB floor that no code compared against. Either
        a floor is enforced or it is not claimed.
        """
        src = inspect.getsource(phase_flip_boot)
        # Matched against the source as WRITTEN, not as rendered: the old text
        # was split across adjacent string literals ("... GB against " / "the
        # 16 GB floor. ..."), so the phrase a reader sees in the log never
        # appears contiguously in the file. An earlier version of this
        # assertion looked for the rendered phrase and therefore passed against
        # the unfixed module -- a test that cannot fail is not a test.
        # The marker is the LEDGER LINE's own wording, not the string "16 GB
        # floor" anywhere in the file -- the commentary that records why the
        # old promise was hollow legitimately contains that phrase, and an
        # assertion blunt enough to catch the explanation as well as the defect
        # would have to be deleted the first time someone documented it.
        self.assertNotIn(
            "host free after",
            src,
            "the ledger still reports `host free after ... against the 16 GB "
            "floor`: a figure from `free -g` compared against nothing",
        )
        # And the claim that replaced it must describe something real: the
        # module refuses via the budget owner rather than shrinking.
        self.assertIn("host available after", src)
        self.assertIn("REFUSES rather than shrinking", src)

    # -- the admission does its job ----------------------------------------

    def test_an_over_commitment_is_refused_by_name(self):
        """Every post priced in the refusal, so an operator knows what to lower."""
        clear_registered_posts()
        with self.assertRaises(ValueError) as ctx:
            check_and_register_pinned_post(
                name="phase-flip staging pin (KV half)",
                flag="--phase-flip-*",
                requested_bytes=400 * GIB,
                reserve_bytes=10 * GIB,
            )
        msg = str(ctx.exception)
        self.assertIn("phase-flip staging pin (KV half)", msg)
        self.assertIn("--phase-flip-*", msg)

    def test_the_pin_is_summed_with_the_other_posts_not_weighed_alone(self):
        """#547 in one assertion: jointly impossible, individually plausible."""
        available = 40 * GIB
        total = 118 * GIB
        hicache = PinnedHostPost(
            name="HiCache staging host tier", flag="--hicache-size", nbytes=25 * GIB
        )
        phase_pin = PinnedHostPost(
            name="phase-flip staging pin (KV half)",
            flag="--phase-flip-*",
            nbytes=20 * GIB,
        )
        # Each ALONE fits under a 10 GiB reserve.
        self.assertIsNone(
            joint_pinned_host_error([hicache], total, available, 10 * GIB)
        )
        self.assertIsNone(
            joint_pinned_host_error([phase_pin], total, available, 10 * GIB)
        )
        # TOGETHER they cannot. This is the sum that was never taken.
        err = joint_pinned_host_error([hicache, phase_pin], total, available, 10 * GIB)
        self.assertIsNotNone(
            err, "two pinned posts that jointly exceed the machine were admitted"
        )
        self.assertIn("phase-flip staging pin (KV half)", err)
        self.assertIn("HiCache staging host tier", err)

    # -- the registered-but-never-allocated hazard --------------------------

    def test_a_failed_allocation_takes_its_post_back(self):
        """The module's own named hazard, asserted on THIS producer's shape.

        Every path below the admission in the pin builder can raise --
        `build_kv_host_pool`, the hybrid entry pair, `HostPoolGroup`. A post
        left behind by one of those would be credited back to the next
        admission as though its bytes were resident, which is the registry
        waving through the over-commitment it exists to refuse.
        """
        src = inspect.getsource(phase_flip_boot)
        # The unregister must live in the failure path of the pin builder.
        self.assertIn(
            "unregister_pinned_post",
            src,
            "a post is registered before an allocation that can fail, and "
            "nothing takes it back",
        )
        # And it must be reachable from the except block, not merely present.
        after_admit = src.split("check_and_register_pinned_post", 1)[1]
        except_idx = after_admit.find("except Exception as exc:")
        self.assertGreater(except_idx, 0, "the pin builder's except block moved")
        self.assertIn(
            "unregister_pinned_post",
            after_admit[except_idx : except_idx + 1500],
            "the failure path of the pin allocation does not take the post back",
        )

    def test_taking_a_post_back_restores_the_budget(self):
        """Behavioural: register, unregister, and the bytes are free again."""
        clear_registered_posts()
        pinned_host_budget.register_pinned_post(
            PinnedHostPost(name="doomed", flag="--x", nbytes=30 * GIB)
        )
        self.assertEqual(len(registered_posts()), 1)
        pinned_host_budget.unregister_pinned_post("doomed")
        self.assertEqual(
            [p.name for p in registered_posts()],
            [],
            "the post survived its own failed allocation and will be credited "
            "back to the next admission as if it were resident",
        )

    def test_both_halves_are_named_posts(self):
        """#871 added a SECOND pinned pool; an unpriced post is what the ledger
        exists to prevent, so the mamba half must be named too."""
        src = inspect.getsource(phase_flip_boot)
        self.assertIn("phase-flip staging pin (KV half)", src)
        self.assertIn("phase-flip staging pin (MAMBA half)", src)

    def test_the_admission_precedes_the_allocation(self):
        """Order is the whole value: admitting after pinning cannot refuse.

        The KV half is admitted BEFORE `build_kv_host_pool`. (The mamba half is
        registered after its own allocation, by necessity -- its per-slot size
        does not exist until the pool is built -- and that asymmetry is stated
        at the call site.)
        """
        src = inspect.getsource(phase_flip_boot)
        admit = src.find("check_and_register_pinned_post")
        alloc = src.find("kv_host = build_kv_host_pool")
        self.assertGreater(admit, 0)
        self.assertGreater(alloc, 0)
        self.assertLess(
            admit,
            alloc,
            "the pin is admitted only after it has already been allocated, "
            "which cannot refuse anything",
        )


if __name__ == "__main__":
    unittest.main()
