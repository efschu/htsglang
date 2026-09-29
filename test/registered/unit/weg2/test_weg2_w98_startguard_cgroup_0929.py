# SPDX-License-Identifier: Apache-2.0
"""W98 z30w (29.09.): the launch guard graded the HOST's MemFree inside Docker.

Boot z30w-park 1 (08:22:13Z, evidence
boot_weg2_dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutz30wparkbar1dauer09290819_*)
was torn down healthy while loading:

    W98 Weg2HostRateLatched: cushion=1.42 GiB BELOW the floor 1.50 while shmem
    is STILL RISING (now=64.28 GiB, shmem=48.26, headroom=31.62 GiB <= relevance
    32.00)

The FREE-POOL-ABSORBS exit (fnFL2 v14) did not take because ``free_gib`` was
``/proc/meminfo MemFree`` -- inside Docker (no lxcfs) that is the HOST's, 34 GiB
at launch against 58 on z30u, the difference foreign page cache while host
MemAvailable stayed above 100. The container's own ceiling (``--memory 84g``)
still had ~19.7 GiB of room. The headroom was also taken against the recorded
host mark 95.90, not against the 84 GiB where this cgroup's OOM fires.

CT999 (memory.max ``max``, lxcfs meminfo) must decide exactly as before:
xsn27 still latches.
"""

import inspect
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

HOST_MARK = 95.90
DOCKER_MAX = 84.0


def _feed(lat, pr, nonreclaim, cushion, shmem):
    """Two ticks with shmem rising by 1 GiB; returns the second tick's line."""
    free, src, ceiling = hl.latch_free_pool(pr)
    lat.observe(0.0, nonreclaim, cushion_gib=cushion, shmem_gib=shmem - 1.0,
                free_gib=free, ceiling_gib=ceiling, free_source=src)
    return lat.observe(0.5, nonreclaim, cushion_gib=cushion, shmem_gib=shmem,
                       free_gib=free, ceiling_gib=ceiling, free_source=src)


class TheZ30wReadingDoesNotLatch(CustomTestCase):
    #: z30w-park 1 at 08:22:13Z; current = nonreclaim + reclaimable file
    #: (cushion 1.42 is the page cache beside shmem), host MemFree < 3.
    PR = {"max_gib": DOCKER_MAX, "current_gib": 65.70, "memavail_gib": 60.0,
          "memfree_gib": 2.10}

    def test_cgroup_room_absorbs_the_write(self):
        lat = hl.RateLatch(reap_mark_gib=HOST_MARK)
        line = _feed(lat, self.PR, 64.28, 1.42, 48.26)
        self.assertFalse(lat.latched, line)
        self.assertIn("FREE-POOL-ABSORBS", line)
        self.assertIn("cgroup room", line)
        # the headroom is against the cgroup ceiling, not the host mark
        self.assertIn("mark=84.00", line)

    def test_the_old_wiring_is_what_tore_it_down(self):
        """The same reading fed the way 50ae2014b0 fed it: host MemFree, no ceiling."""
        lat = hl.RateLatch(reap_mark_gib=HOST_MARK)
        lat.observe(0.0, 64.28, cushion_gib=1.42, shmem_gib=47.26, free_gib=2.10)
        line = lat.observe(0.5, 64.28, cushion_gib=1.42, shmem_gib=48.26,
                           free_gib=2.10)
        self.assertTrue(lat.latched)
        self.assertIn("headroom=31.62", line)


class RealDeathsStillLatch(CustomTestCase):
    def test_docker_shape_at_the_ceiling_latches(self):
        """Cgroup room gone (0.4 GiB), cushion spent, shmem still rising."""
        pr = {"max_gib": DOCKER_MAX, "current_gib": 83.60, "memavail_gib": 40.0,
              "memfree_gib": 30.0}
        lat = hl.RateLatch(reap_mark_gib=HOST_MARK)
        line = _feed(lat, pr, 82.50, 0.50, 60.0)
        self.assertTrue(lat.latched, line)
        self.assertIn("W98 Weg2HostRateLatched", line)
        self.assertIn("headroom=1.50", line)

    def test_ceiling_pulls_a_far_reading_inside_the_relevance_bound(self):
        """Against 95.90 the headroom 35.9 would only be NOTED; against 84 it is 24."""
        pr = {"max_gib": DOCKER_MAX, "current_gib": 83.0, "memavail_gib": 50.0,
              "memfree_gib": 1.0}
        lat = hl.RateLatch(reap_mark_gib=HOST_MARK)
        line = _feed(lat, pr, 60.0, 1.0, 40.0)
        self.assertTrue(lat.latched, line)
        self.assertIn("headroom=24.00", line)

    def test_ct999_xsn27_shape_is_unchanged(self):
        """memory.max 'max' -> lxcfs MemFree, host mark: xsn27 latched, still does."""
        pr = {"max_gib": None, "current_gib": 91.8, "memavail_gib": 5.0,
              "memfree_gib": 0.9}
        free, src, ceiling = hl.latch_free_pool(pr)
        self.assertEqual((free, src, ceiling), (0.9, "MemFree", None))
        lat = hl.RateLatch(reap_mark_gib=HOST_MARK)
        line = _feed(lat, pr, 80.0, 0.01, 60.0)
        self.assertTrue(lat.latched, line)
        self.assertIn("headroom=15.90", line)


class TheReaderCarriesCeilingAndAvailable(CustomTestCase):
    def _read(self, max_text):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "memory.max"), "w") as f:
                f.write(max_text + "\n")
            g = int(hl.GIB)
            with open(os.path.join(d, "memory.current"), "w") as f:
                f.write(f"{70 * g}\n")
            with open(os.path.join(d, "memory.stat"), "w") as f:
                f.write(f"anon {30 * g}\nshmem {35 * g}\nfile {37 * g}\n"
                        f"inactive_file {g}\nactive_file {g}\n")
            mi = os.path.join(d, "meminfo")
            with open(mi, "w") as f:
                f.write("MemTotal:  131072000 kB\nMemFree:  2097152 kB\n"
                        "MemAvailable:  62914560 kB\n")
            return hl.read_cgroup_pressure(d, meminfo_path=mi)

    def test_finite_max(self):
        pr = self._read(str(int(DOCKER_MAX * hl.GIB)))
        self.assertAlmostEqual(pr["max_gib"], DOCKER_MAX)
        self.assertAlmostEqual(pr["memfree_gib"], 2.0)
        self.assertAlmostEqual(pr["memavail_gib"], 60.0)
        free, src, ceiling = hl.latch_free_pool(pr)
        self.assertAlmostEqual(free, 14.0)  # 84 - 70, smaller than 60
        self.assertEqual(ceiling, DOCKER_MAX)
        self.assertIn("cgroup room", src)

    def test_max_literal_stays_none(self):
        pr = self._read("max")
        self.assertIsNone(pr["max_gib"])
        self.assertEqual(hl.latch_free_pool(pr)[1], "MemFree")


class BothLatchSitesUseThePool(CustomTestCase):
    def test_launch_guard(self):
        from sglang.srt.weg2 import launcher
        src = inspect.getsource(launcher.LaunchGuard._run)
        self.assertIn("host_ledger.latch_free_pool(pr)", src)
        self.assertIn("ceiling_gib=ceiling", src)

    def test_front_loop(self):
        from sglang.srt.weg2 import front as fr
        src = inspect.getsource(fr)
        self.assertIn("host_ledger.latch_free_pool(_pr_fast)", src)
        self.assertIn("ceiling_gib=_ceiling", src)


if __name__ == "__main__":
    unittest.main()
