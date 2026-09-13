# SPDX-License-Identifier: Apache-2.0
"""#1358 -- the bounce term is charged PER LANE, and the lane checks the count.

MEASURED THREE WAYS on boot weg2xsn28, all agreeing on 7.044 GiB of assemble
buffer while the ARM line charged 2.160:

    filesystem, 0 holders at teardown
        /dev/shm/weg2-xchg-<epoch>/bounce.bin.{c0,c1,p1,p2,p4} = 5 x 1.409
    cgroup sampler, d_shmem across the source leg      +7.043 GiB
    the lane's OWN HOST-SLOT lines                4.226 + 2.818 = 7.044 GiB

COUNTER-PROOF that the instrument is sound, same method same boot: the host
ring summed to 46.40 GiB against `host_weights=46.40` -- exact. The ledger
priced the ring right and the bounce wrong, so it is not a tmpfs or
sparse-file artefact.

ALL FIVE ARE CONCURRENT: the two legs overlap 115 s of 120 s. So this is a sum
and not a peak, and reducing the count would be a lane change rather than an
accounting one.

NO LITERAL ANYWHERE. `n_lanes` rides with the term's INPUTS through
`publish_terms`, so the rank recomputes the same total instead of holding one
it cannot check -- the rule that tuple already existed for -- and the leg
driver checks the priced count against `group_descs_by_pair`, which IS the
lane enumeration.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import xchg_bounce as xb
from sglang.test.test_utils import CustomTestCase

G = 1024 ** 3
WIDEST = 756323776          # gdncov INT8, 721.3 MiB
XSN28 = dict(bytes_per_direction=29119878266, n_layers=64,
             widest_layer_bytes=WIDEST, pairs=3, depth=1,
             slot_bytes=134217728)


class TheTermIsChargedPerLane(CustomTestCase):
    def test_one_lane_is_byte_identical_to_before(self):
        t = xb.bounce_terms(**XSN28)
        self.assertEqual(t.n_lanes, 1)
        self.assertEqual(t.total_bytes, t.buffer_bytes + t.staging_bytes)

    def test_the_xsn28_replay_gives_five_lanes_and_7044_GiB_of_buffer(self):
        """THE ORDERED NUMBER, as a number."""
        t = xb.bounce_terms(**dict(XSN28, n_lanes=5))
        self.assertAlmostEqual(t.buffer_bytes * t.n_lanes / G, 7.044, places=2)
        self.assertAlmostEqual(t.buffer_bytes / G, 1.409, places=2)

    def test_a_six_lane_cut_gives_845(self):
        t = xb.bounce_terms(**dict(XSN28, n_lanes=6))
        self.assertAlmostEqual(t.buffer_bytes * t.n_lanes / G, 8.453, places=2)

    def test_the_lane_count_is_not_a_literal_in_the_term(self):
        import inspect

        src = "\n".join(
            ln for ln in inspect.getsource(xb.bounce_terms).splitlines()
            if not ln.strip().startswith("#"))
        for lit in ("= 5", "=5", "* 5"):
            self.assertNotIn(lit, src)

    def test_the_count_rides_with_the_inputs_and_round_trips(self):
        """A rank must recompute the total, never inherit one it cannot check."""
        self.assertIn("n_lanes", xb._TERM_FIELDS)
        for n in (1, 5, 6):
            with self.subTest(n_lanes=n):
                t = xb.bounce_terms(**dict(XSN28, n_lanes=n))
                back = xb.read_published_terms(xb.publish_terms(t))
                self.assertEqual(int(back.n_lanes), n)
                self.assertEqual(int(back.total_bytes), int(t.total_bytes))


class TheLaneChecksTheLedgersCount(CustomTestCase):
    def test_the_leg_driver_compares_against_its_own_enumeration(self):
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        i = src.index("group_descs_by_pair(descs)")
        window = src[max(0, i - 1500):i + 1500]
        self.assertIn("n_lanes", window)
        self.assertIn("W68 Weg2XchgPlanDisagree", window)

    def test_the_refusal_is_raised_before_the_first_allocation(self):
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        # FROM the guard forward: an earlier `run_bounce_leg` mention exists
        # in this module (a docstring), and searching before the guard would
        # have compared against the wrong occurrence -- the same trap as
        # matching a marker inside the prose that describes it (#1362
        # [22-fix3]).
        guard = src.index("the ledger priced ")
        alloc = src.index("bx.run_bounce_leg(", guard)
        self.assertLess(guard, alloc,
                        "the count must be checked BEFORE a buffer is created")


class TheCushionIsReservedNotOnlyPredicted(CustomTestCase):
    """weg2xsn28 armed with 9.11 GiB of room and latched W98 nine seconds after
    `serving`, short 0.51 GiB -- before any flip. An arm cannot lose at the run
    moment what it never counted at the launch moment."""

    def test_the_floor_is_imported_not_restated(self):
        import inspect

        src = inspect.getsource(hl.cushion_floor_reserve_gib)
        self.assertIn("RATE_LATCH_CUSHION_FLOOR_GIB", src)
        self.assertNotIn("1.5", src.split('"""')[-1])

    def test_the_reserve_is_the_floor_plus_what_sits_in_shmem(self):
        self.assertAlmostEqual(hl.cushion_floor_reserve_gib(0.0),
                               hl.RATE_LATCH_CUSHION_FLOOR_GIB, places=6)
        self.assertAlmostEqual(hl.cushion_floor_reserve_gib(7.044), 8.544,
                               places=3)

    def test_a_negative_bounce_cannot_shrink_the_floor(self):
        self.assertAlmostEqual(hl.cushion_floor_reserve_gib(-9.0),
                               hl.RATE_LATCH_CUSHION_FLOOR_GIB, places=6)

    def test_the_guard_and_the_arm_read_ONE_threshold(self):
        """Two spellings of one threshold is how an arm funds a boot the guard
        then tears down."""
        import inspect

        self.assertIn("RATE_LATCH_CUSHION_FLOOR_GIB",
                      inspect.getsource(hl.RateLatch.observe))


if __name__ == "__main__":
    unittest.main()
