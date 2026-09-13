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


class TheLaneCountIsMeasuredAndKeyedOnTheCut(CustomTestCase):
    """#1358 producer -- the count comes from a boot, not from a derivation.

    The lane set is built at the RANK at runtime by `group_descs_by_pair` over
    descs carrying pointers. It does not exist at the arm: the xchg census
    carries `cards` and `waves` and no (src,dst) structure (checked), and
    deriving it from the P-cut would be a second expression of an enumeration
    the lane already owns -- the defect this ticket removes, not repeats.

    So it is LEARNED, like Sigma H, from the boot's own HOST-SLOT lines: the
    distinct `path=` among `event=alloc`. Measured on weg2xsn28 by the #1358
    reader: 5 -- bounce.bin.{c0,c1,p1,p2,p4}.

    KEYED ON THE CUT, NOT THE FORM KEY, and that decides xsn29: text-only
    (#1356) moves the form key and leaves the cut alone, so this seed carries.
    """

    CUT = "39,13,12"

    def test_the_xsn28_seed_is_five_and_names_its_boot(self):
        n, prov = hl.resolve_xchg_lanes(hl.xchg_cut_key(self.CUT))
        self.assertEqual(n, 5)
        for token in ("WEG2-XCHG-LANES", "lanes=5", "boot=weg2xsn28",
                      "source=measured", "bounce.bin.c0"):
            self.assertIn(token, prov, f"the provenance omits {token}")

    def test_an_unmeasured_cut_is_refused_by_name_not_defaulted(self):
        """A guessed lane count is the 4.88 GiB under-charge with a comment."""
        # #1362 [bootstrap]: no longer a refusal. An unrecorded cut is a first
        # boot -- priced at the region's worst case, named as such, and the
        # line still lists the cuts that ARE measured so the reader can see
        # which case this is. W102 survives for the leg driver, where a
        # RECORDED cut that disagrees with what the leg creates is still
        # refused before the first allocation.
        n, m = hl.resolve_xchg_lanes(hl.xchg_cut_key("40,12,12"))
        self.assertIn("WORST-CASE", m)
        self.assertIn("bootstrap", m)
        self.assertIn("39,13,12", m, "the line must name the cuts it knows")
        self.assertGreaterEqual(n, 5, "the bootstrap count must not under-charge")

    def test_the_key_is_the_cut_and_not_the_form(self):
        """Text-only changes the form key; the cut and so the lanes are the same."""
        self.assertEqual(hl.xchg_cut_key(self.CUT), hl.xchg_cut_key(self.CUT))
        self.assertNotEqual(hl.xchg_cut_key(self.CUT),
                            hl.xchg_cut_key("40,12,12"))
        self.assertIn("pp=39,13,12", hl.xchg_cut_key(self.CUT))
        self.assertNotIn("vision", hl.xchg_cut_key(self.CUT))
        self.assertNotIn("multimodal", hl.xchg_cut_key(self.CUT))

    def test_the_arm_reads_it_and_the_refusal_is_a_ledger_refusal(self):
        self.assertTrue(
            issubclass(hl.Weg2XchgLanesUnmeasured, hl.Weg2HostLedgerRefused))

    def test_the_launcher_asks_for_it_rather_than_defaulting(self):
        import inspect

        from sglang.srt.weg2 import launcher as lc

        src = "\n".join(ln for ln in inspect.getsource(lc).splitlines()
                        if not ln.strip().startswith("#"))
        self.assertIn("resolve_xchg_lanes(", src)
        self.assertIn("xchg_cut_key(", src)
        i = src.index("xchg_bounce_terms_for_arm(\n")
        self.assertIn("_lane_n", src[i:i + 300],
                      "the arm must pass the MEASURED count, not a default")

    def test_the_seed_matches_what_the_reader_measures(self):
        """THE RATCHET: the recorded number and the boot's own lines agree.

        If they ever diverge, one of them is wrong and this says so here
        instead of at the next arm.
        """
        rec = hl.XCHG_LANES_BY_CUT[hl.xchg_cut_key(self.CUT)]
        self.assertEqual(int(rec["lanes"]), len(rec["paths"]),
                         "the recorded count and the recorded paths disagree")
        self.assertEqual(
            sorted(rec["paths"]),
            ["bounce.bin.c0", "bounce.bin.c1", "bounce.bin.p1",
             "bounce.bin.p2", "bounce.bin.p4"],
            "the seed is not weg2xsn28's measured file set")
