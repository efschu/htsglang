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
DK5_CHUNKS = 8
DK5_STORE_MIN_GIB = 8.0


def _choose(**over):
    kw = dict(
        store_min_gib=DK5_STORE_MIN_GIB,
        weight_chunks=DK5_CHUNKS,
        cg_current_bytes=DK5_CG_CURRENT_B,
        cg_reclaimable_bytes=DK5_CG_RECLAIMABLE_B,
        cg_ceiling_bytes=DK5_MEMTOTAL_B,
        cg_ceiling_source="test: lxcfs MemTotal fallback",
        cg_oom_kill=18,
    )
    kw.update(over)
    return host_ledger.choose(DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, **kw)


class TestTheReclaimableTerm(CustomTestCase):
    def test_dk5_launch_readings_fund_the_arm_that_boot_actually_ran(self):
        # The regression fix 5 introduced: at these readings every arm refused,
        # so the branch could not launch from the state it last launched from.
        arm, store, lines = _choose()
        self.assertEqual((arm.s_gb, arm.m_mib), (1, 1200))
        self.assertEqual(store, 8.0)
        # base = 118.05 (ceiling) - 14.99 (non-reclaimable) - 10 (CLI) = 93.05,
        # 6.17 GiB looser than fix 5's 86.89 and still 10.5 GiB tighter than
        # the meminfo arm -- the cgroup still binds.
        self.assertAlmostEqual(arm.terms["base_gib"], 93.05, delta=0.05)
        self.assertAlmostEqual(arm.run_leftover_gib, 8.16, delta=0.05)
        self.assertIn("cgroup", arm.terms["base_source"])
        # M=2400 stays refused: the correction re-prices, it does not open the
        # ladder's top arm by fiat.
        self.assertTrue(any("S=1 M=2400" in ln and "refused" in ln for ln in lines))

    def test_charging_the_whole_reading_refuses_that_same_launch(self):
        # Exactly fix 5's arithmetic, reachable through the same seam: with no
        # memory.stat the ledger charges everything and REFUSES rather than
        # inventing a reclaimable share.
        with self.assertRaises(host_ledger.Weg2HostLedgerRefused):
            _choose(cg_reclaimable_bytes=None)

    def test_the_terms_line_names_the_reclaimable_term_by_name(self):
        _arm, _store, lines = _choose()
        terms = lines[0]
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
            weight_chunks=DK5_CHUNKS,
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
            weight_chunks=DK5_CHUNKS,
            cg_current_bytes=DK5_CG_CURRENT_B,
            cg_reclaimable_bytes=DK5_CG_CURRENT_B * 4,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        self.assertEqual(arm.terms["cg_nonreclaim_gib"], 0.0)

    def test_base_is_still_the_tighter_of_the_two_readings(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            weight_chunks=DK5_CHUNKS,
            cg_current_bytes=0,
            cg_reclaimable_bytes=0,
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
    def test_the_predicted_origin_is_the_non_reclaimable_reading(self):
        common = dict(
            weight_chunks=DK5_CHUNKS,
            cg_current_bytes=DK5_CG_CURRENT_B,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        cached = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            cg_reclaimable_bytes=DK5_CG_RECLAIMABLE_B, **common
        )
        charged = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            cg_reclaimable_bytes=0, **common
        )
        # The only difference between the two is cache the kernel would hand
        # back; the predicted peak must move by exactly that and no more.
        self.assertAlmostEqual(
            charged.predicted_run_peak_gib(8.0) - cached.predicted_run_peak_gib(8.0),
            DK5_CG_RECLAIMABLE_B / GIB,
            delta=0.01,
        )
        self.assertAlmostEqual(
            cached.predicted_run_peak_gib(8.0) - cached.terms["cg_nonreclaim_gib"],
            cached.predicted_run_peak_gib(0.0) - cached.terms["cg_nonreclaim_gib"] + 8.0,
            delta=1e-6,
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
        _arm, _store, lines = _choose()
        advisory = [ln for ln in lines if "RUN-PEAK ADVISORY" in ln][0]
        self.assertIn("non-reclaimable", advisory)
        self.assertIn("95.90 GiB", advisory)

    def test_no_cgroup_sample_still_states_the_absence(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, weight_chunks=DK5_CHUNKS
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
