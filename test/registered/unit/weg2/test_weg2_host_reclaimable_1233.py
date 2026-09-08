# SPDX-License-Identifier: Apache-2.0
"""Weg 2 (#1233, fix 6): the host budget's denominator is the NON-RECLAIMABLE
part of ``memory.current``, not the whole reading.

Fix 5 swapped one wrong denominator for another in the opposite direction.  It
was right that the reaper watches the cgroup and not /proc/meminfo, and wrong
about WHAT it watches: ``memory.current`` includes the page cache, which the
kernel hands back under pressure instead of killing for.  Charging it as spent
made the ledger refuse room that is not occupied.

Two measured consequences, both pinned below:

1. AT BOOT weg2dk5's OWN LAUNCH READINGS the whole ladder refuses -- the branch
   cannot launch from the state its last boot launched from.  Of that boot's
   21.16 GiB ``memory.current``, 6.17 GiB was reclaimable page cache
   (memts row 21:05:49Z: ``cached_kb`` 9,859,520 minus ``shmem_kb`` 3,393,080 =
   ``pagecache_ex_shmem_kb`` 6,466,440).  Not charging it puts S=1 M=1200 back
   on the ladder as the first fundable arm -- the arm that boot actually ran.
2. THE RUN-PEAK ADVISORY compared an inflated origin (``memory.current``
   including cache) against a reap watermark that was almost pure
   non-reclaimable memory (at the 21:15:30Z reap row ``pagecache_ex_shmem_kb``
   is 28,916 kB = 0.03 GiB of the 95.93 GiB reading).  Origin and watermark are
   now stated in the same currency.

WHAT IS RECLAIMABLE, with the cgroup-v2 semantics MEASURED on this box rather
than recalled (2026-09-07, before the fix; see
:data:`host_ledger.CGROUP_V2_FILE_INCLUDES_SHMEM_PROOF`):

* ``memory.stat file`` = 51,171,528,704 B and /proc/meminfo ``Cached`` =
  49,972,196 kB = 51,171,528,704 B -- EQUAL to the byte, so v2's ``file`` counts
  the whole page cache, shmem included;
* ``memory.stat shmem`` = 20,425,609,216 B and /proc/meminfo ``Shmem`` =
  19,946,884 kB = 20,425,609,216 B -- equal to the byte as well;
* /proc/meminfo ``SwapTotal`` = 0: shmem cannot be evicted at all on this box.

So the reclaimable term is ``(file - shmem) + slab_reclaimable`` and NEVER
``current - file``: the latter would hand back the tmpfs page store -- the
carrier itself -- as if it were free.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no server; every cgroup fact is passed
in or written into a temporary fake cgroup tree.
"""

import inspect
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger
from sglang.test.test_utils import CustomTestCase

GIB = host_ledger.GIB

# ------------------------------------------------------- MEASURED, boot weg2dk5
#: Front log 21:05:49Z ``WEG2-HOST-LEDGER TERMS memtotal=118.05 memavail=103.58``
#: and the memts row of the same second, /spinning/gpu-arb/memts_weg2_weg2dk5.csv:
#: ``cg_current_b=22719148032, cached_kb=9859520, shmem_kb=3393080,
#: pagecache_ex_shmem_kb=6466440, oom_kill=18``.
DK5_MEMTOTAL_B = 126_751_866_880
DK5_MEMAVAIL_B = 111_196_077_056
DK5_CG_CURRENT_B = 22_719_148_032
#: The reclaimable share of that reading: page cache MINUS shmem.  The memts
#: sampler carries no slab column, so ``slab_reclaimable`` is charged as spent
#: here -- a conservative reading of a term that was 0.73 GiB on this box.
DK5_CG_RECLAIMABLE_B = 6_466_440 * 1024
#: C19 (ring rebase 0908): the host weights term is the measured per-card ring
#: table (Sigma H / Sigma image_P), not a chunk count.  Same figures the s3s4
#: and ring_ledger suites pin.
DK5_RING_BYTES = 32964 * 1024 * 1024
DK5_RING_SPAN1_BYTES = 29912 * 1024 * 1024
RING_KW = dict(ring_bytes=DK5_RING_BYTES, ring_span1_bytes=DK5_RING_SPAN1_BYTES)
DK5_STORE_MIN_GIB = 8.0


def _choose(**over):
    kw = dict(
        store_min_gib=DK5_STORE_MIN_GIB,
        **RING_KW,
        cg_current_bytes=DK5_CG_CURRENT_B,
        reclaimable_bytes=DK5_CG_RECLAIMABLE_B,
        cg_ceiling_bytes=DK5_MEMTOTAL_B,
        cg_ceiling_source="test: lxcfs MemTotal fallback",
        cg_oom_kill=18,
    )
    kw.update(over)
    return host_ledger.choose(DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, **kw)


def _lines(**over):
    """The ladder's lines whether it funds an arm or refuses.

    FIX 8 (boot weg2dk7): at weg2dk5's readings the MEASURED dormant image
    (38.63 GiB against the 28.83 GiB weight-tag census this file was written
    against) refuses every arm, and the TERMS and RUN-PEAK lines are printed
    with the refusal exactly as they are with a choice -- so the evidence these
    tests read is at the same place either way.
    """
    try:
        _arm, _store, lines = _choose(**over)
        return lines
    except (host_ledger.Weg2HostLedgerRefused, host_ledger.Weg2HostRunPeakRefused) as e:
        return str(e).splitlines()


class TestTheReclaimableTerm(CustomTestCase):
    def test_dk5_launch_readings_still_give_the_base_this_term_funds(self):
        # UPDATED BY FIX 8, and the update is a measurement, not a taste: this
        # test used to assert that these readings FUND S=1 M=1200 with an 8 GiB
        # store.  Boot weg2dk7 then measured the dormant image at 38.63 GiB
        # against the 28.83 GiB weight-tag census the ledger charged, +9.80 GiB
        # on one image, and at that price weg2dk5's own launch state funds no
        # arm at its own store floor.  What fix 6 is about -- the DENOMINATOR --
        # is unchanged and is what this test pins now.
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            **RING_KW,
            cg_current_bytes=DK5_CG_CURRENT_B,
            reclaimable_bytes=DK5_CG_RECLAIMABLE_B,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        # base = 118.05 (ceiling) - 14.99 (non-reclaimable) - 10 (CLI) = 93.05,
        # 6.17 GiB looser than fix 5's 86.89 and still 10.5 GiB tighter than
        # the meminfo arm -- the cgroup still binds.
        self.assertAlmostEqual(arm.terms["base_gib"], 93.05, delta=0.05)
        self.assertIn("cgroup", arm.terms["base_source"])
        # RE-DERIVED ON THE RING (rebase 0908); the delta is the ring's whole
        # reason for existing, not a fitted number.  Fix 8 charged the run
        # moment the dormant image (38.63) PLUS the flip transient (9.97) =
        # 48.60 GiB; C19 charges Sigma H (32.19) ONCE instead, because the
        # region is preallocated and the legs copy through it.  The run
        # leftover rises by EXACTLY 48.60 - 32.19 = 16.41 GiB, from fix 8's
        # -1.64 (no arm) to 14.77 (funded) -- FLIPCOST A1-3 as arithmetic.
        ring_saving_gib = (38.63 + 9.97) - 32.19
        self.assertAlmostEqual(
            arm.run_leftover_gib,
            (8.16 - arm.terms["image_extra_p_gib"]) + ring_saving_gib,
            delta=0.05,
        )
        lines = _lines()
        self.assertTrue(any("S=1 M=2400" in ln and "refused" in ln for ln in lines))

    def test_charging_the_whole_reading_costs_exactly_the_cache_it_charges(self):
        # Exactly fix 5's arithmetic, reachable through the same seam: with no
        # memory.stat the ledger charges everything rather than inventing a
        # reclaimable share.  Under fix 8 BOTH readings refuse (the measured
        # image is 9.80 GiB bigger than the census), so the contrast this test
        # exists for is stated where it survives -- as a size, not a verdict:
        # the run leftover moves by EXACTLY the reclaimable share, 6.17 GiB.
        # RING REBASE 0908: under fix 8 BOTH readings refused; on the ring both
        # FUND (Sigma H 32.19 replaces image + transient 48.60, 16.41 cheaper).
        # The VERDICT has flipped twice and is not what this test pins -- the
        # comment above already said so.  The SIZE is the pin and it is
        # invariant under both changes.  The refusal behaviour still has a
        # home: test_weg2_fix7_wires drives it from files through the seam.
        common = dict(
            **RING_KW, cg_current_bytes=DK5_CG_CURRENT_B,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        cached = host_ledger.price(DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
                                   reclaimable_bytes=DK5_CG_RECLAIMABLE_B, **common)
        charged = host_ledger.price(DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
                                    reclaimable_bytes=None, **common)
        self.assertAlmostEqual(
            cached.run_leftover_gib - charged.run_leftover_gib,
            DK5_CG_RECLAIMABLE_B / GIB, delta=0.01,
        )

    def test_the_terms_line_names_the_reclaimable_term_by_name(self):
        terms = [ln for ln in _lines() if "WEG2-HOST-LEDGER TERMS" in ln][0]
        self.assertIn("reclaimable=", terms)
        self.assertIn("6.17 GiB", terms)          # the measured share
        self.assertIn("non-reclaimable", terms)
        self.assertIn("14.99 GiB", terms)         # what is actually charged
        # the FORMULA, not just the number: an auditable denominator names how
        # it was derived, including the v2 semantics that make shmem spent.
        self.assertIn("file - shmem", terms)
        self.assertIn("slab_reclaimable", terms)

    def test_an_unreadable_memory_stat_is_named_never_priced_as_zero_cache(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            **RING_KW,
            cg_current_bytes=DK5_CG_CURRENT_B,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        self.assertIsNone(arm.terms["cg_reclaimable_gib"])
        self.assertAlmostEqual(arm.terms["cg_nonreclaim_gib"], 21.16, delta=0.05)
        self.assertIn("memory.stat unreadable", arm.terms["base_source"])

    def test_shmem_is_spent_because_v2_file_already_counts_it(self):
        # The trap the docstring measures: `current - file` would return the
        # page store's own tmpfs as free.  file=10 GiB of which shmem=6 GiB,
        # slab_reclaimable=1 GiB -> reclaimable is 5 GiB, not 11.
        stat = {
            "file": 10 * 2**30,
            "shmem": 6 * 2**30,
            "slab_reclaimable": 1 * 2**30,
            "anon": 3 * 2**30,
            "unevictable": 0,
        }
        self.assertEqual(host_ledger.cg_reclaimable_bytes(stat), 5 * 2**30)

    def test_a_partial_memory_stat_is_none_not_a_guess(self):
        self.assertIsNone(host_ledger.cg_reclaimable_bytes({"file": 1 << 30}))
        self.assertIsNone(host_ledger.cg_reclaimable_bytes({}))

    def test_the_reclaimable_share_can_never_exceed_the_reading(self):
        # A stat read a moment after current (they are two files) must not turn
        # into a negative occupancy, i.e. free memory this ledger never had.
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            **RING_KW,
            cg_current_bytes=DK5_CG_CURRENT_B,
            reclaimable_bytes=DK5_CG_CURRENT_B * 4,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        self.assertEqual(arm.terms["cg_nonreclaim_gib"], 0.0)

    def test_base_is_still_the_tighter_of_the_two_readings(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            **RING_KW,
            cg_current_bytes=0,
            reclaimable_bytes=0,
            cg_ceiling_bytes=DK5_MEMTOTAL_B * 4,
        )
        self.assertAlmostEqual(arm.terms["base_gib"], 103.56, delta=0.05)
        self.assertIn("meminfo", arm.terms["base_source"])


class TestReadCgroupReadsTheStat(CustomTestCase):
    def _tree(self, d, stat=True):
        files = {
            "memory.current": "22719148032\n",
            "memory.peak": "106193485824\n",
            "memory.max": "max\n",
            "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 18\n",
        }
        if stat:
            files["memory.stat"] = (
                "anon 12000000000\nfile 10000000000\nkernel 1\nshmem 4000000000\n"
                "unevictable 81920\nslab_reclaimable 500000000\n"
                "slab_unreclaimable 300000000\n"
            )
        for name, text in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)

    def test_the_stat_terms_and_the_derived_reclaimable_come_back(self):
        with tempfile.TemporaryDirectory() as d:
            self._tree(d)
            cg = host_ledger.read_cgroup(d)
        self.assertEqual(cg["file"], 10_000_000_000)
        self.assertEqual(cg["shmem"], 4_000_000_000)
        self.assertEqual(cg["slab_reclaimable"], 500_000_000)
        self.assertEqual(cg["anon"], 12_000_000_000)
        self.assertEqual(cg["unevictable"], 81920)
        self.assertEqual(cg["reclaimable"], 10_000_000_000 - 4_000_000_000 + 500_000_000)

    def test_a_missing_memory_stat_leaves_reclaimable_none(self):
        with tempfile.TemporaryDirectory() as d:
            self._tree(d, stat=False)
            cg = host_ledger.read_cgroup(d)
        self.assertEqual(cg["current"], 22_719_148_032)
        self.assertIsNone(cg["reclaimable"])
        self.assertIsNone(cg["file"])

    def test_a_missing_tree_is_all_none(self):
        cg = host_ledger.read_cgroup("/nonexistent-cgroup-root-1233")
        self.assertIsNone(cg["reclaimable"])
        self.assertIsNone(cg["current"])

    def test_this_box_agrees_with_the_measurement_in_the_docstring(self):
        # Not a re-derivation of the formula: a live CROSS-CHECK that the two
        # files this fix reads still mean what they were measured to mean here
        # (v2 `file` == meminfo Cached, v2 `shmem` == meminfo Shmem).  Skipped
        # rather than guessed where the cgroup is unreadable.
        cg = host_ledger.read_cgroup()
        if cg["file"] is None:
            self.skipTest("no cgroup2 memory.stat on this host")
        mi = host_ledger.read_meminfo()
        self.assertGreaterEqual(cg["file"], cg["shmem"])
        # tolerance: the two files are read microseconds apart under live load.
        self.assertAlmostEqual(cg["shmem"] / GIB, mi["Shmem"] / GIB, delta=1.0)
        self.assertAlmostEqual(cg["file"] / GIB, mi["Cached"] / GIB, delta=1.0)


class TestTheRunPeakAdvisory(CustomTestCase):
    def test_the_predicted_origin_is_the_run_moment_not_the_launch_reading(self):
        # SUPERSEDED BY FIX 8, and this is the whole point of that fix: this
        # test used to assert that the predicted peak moves with the LAUNCH
        # reading (by exactly the reclaimable share).  Boots weg2dk6/dk7 showed
        # that selector is anti-correlated with safety -- the quieter launch
        # bought the bigger arm and produced the tighter boot -- so the origin
        # is now the RUN-moment residual and BOTH of weg2dk5's readings sit
        # below it.  The base still moves by the reclaimable share (pinned
        # above); the PEAK deliberately no longer does.
        common = dict(
            **RING_KW,
            cg_current_bytes=DK5_CG_CURRENT_B,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        cached = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            reclaimable_bytes=DK5_CG_RECLAIMABLE_B, **common
        )
        charged = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            reclaimable_bytes=0, **common
        )
        for arm in (cached, charged):
            self.assertLess(arm.terms["cg_nonreclaim_gib"], arm.terms["run_origin_gib"])
            self.assertIn("RUN-MOMENT RESIDUAL FLOOR", arm.terms["run_origin_source"])
        self.assertAlmostEqual(
            charged.predicted_run_peak_gib(8.0), cached.predicted_run_peak_gib(8.0),
            delta=1e-9,
        )
        # The store is still a term of the sum, one GiB for one GiB.
        self.assertAlmostEqual(
            cached.predicted_run_peak_gib(8.0) - cached.predicted_run_peak_gib(0.0),
            8.0, delta=1e-6,
        )

    def test_the_watermark_is_stated_in_the_same_currency(self):
        # 102,998,904,832 B at the reap row, of which pagecache_ex_shmem_kb =
        # 28,916 kB was reclaimable -- 0.03 GiB.  The watermark was already
        # almost pure non-reclaimable memory; saying so is what makes the
        # comparison legitimate rather than lucky.
        self.assertEqual(
            host_ledger.OBSERVED_REAP_NONRECLAIM_BYTES,
            host_ledger.OBSERVED_REAP_CURRENT_BYTES - 28_916 * 1024,
        )
        advisory = [ln for ln in _lines() if "RUN-PEAK ADVISORY" in ln][0]
        self.assertIn("non-reclaimable", advisory)
        self.assertIn("95.90 GiB", advisory)

    def test_no_cgroup_sample_still_states_the_absence(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, **RING_KW
        )
        self.assertIsNone(arm.predicted_run_peak_gib(8.0))

    def test_the_v2_semantics_are_written_down_where_the_formula_lives(self):
        proof = host_ledger.CGROUP_V2_FILE_INCLUDES_SHMEM_PROOF
        self.assertIn("Cached", proof)
        self.assertIn("Shmem", proof)
        self.assertIn("SwapTotal", proof)
        doc = inspect.getdoc(host_ledger.cg_reclaimable_bytes)
        self.assertIn("file - shmem", doc)
        self.assertIn("slab_reclaimable", doc)


if __name__ == "__main__":
    unittest.main()
