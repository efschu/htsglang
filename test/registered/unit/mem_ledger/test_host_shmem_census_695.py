"""#695: the host-shmem census, against recorded /proc shapes.

The shapes that matter are awkward to produce on demand and easy to record,
so the parser is fed a fixture written from the real thing: the three
``sglang::scheduler`` ranks of the 2026-08-12 PP=3 boot, whose 75.0 GiB of
page-locked ``MAP_SHARED`` memory was invisible to every ledger in the tree.

Three properties are pinned, each one a way the census could be quietly wrong:

1.  **Anonymous shared memory is classified on the SHARED BIT, not the path.**
    The kernel labels ``MAP_SHARED|MAP_ANONYMOUS`` as ``/dev/zero (deleted)``.
    A classifier keying on that string would also catch a private ``/dev/zero``
    mapping, and one keying on "has no path" would miss it entirely.
2.  **Driver mappings are measured and then excluded.** ``/dev/nvidia*`` is
    device/BAR space, not host pages. Charging it would inflate the exact
    number this census exists to make trustworthy.
3.  **The total is Pss, not Rss.** A shmem page mapped twice is charged once to
    the cgroup; an Rss sum double-counts and stops reconciling against
    ``memory.stat``.

CAN-FAIL: classify anon-shared as file-backed and
``test_anon_shared_is_the_dominant_class`` goes red; fold ``driver`` into
``HOST_RAM_CLASSES`` and ``test_driver_mappings_are_measured_but_not_charged``
goes red; sum Rss instead of Pss and ``test_the_total_is_pss_not_rss`` goes
red.
"""

import os
import tempfile
import unittest

from sglang.srt.mem_ledger.host_shmem import (
    CLASS_ANON_SHARED,
    CLASS_DRIVER,
    CLASS_FILE_SHARED,
    CLASS_SHM_NAMED,
    CLASS_SHM_NCCL,
    HostShmemCensus,
    classify_mapping,
    parse_shared_mappings,
    render_host_shmem_line,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GIB = 1 << 30
MIB = 1 << 20


def _vma(start, size_kb, perms, path, pss_kb=None, rss_kb=None):
    """One smaps stanza in the kernel's own format."""
    end = start + size_kb * 1024
    rss = size_kb if rss_kb is None else rss_kb
    pss = rss if pss_kb is None else pss_kb
    return (
        f"{start:012x}-{end:012x} {perms} 00000000 00:01 12345 {path}\n"
        f"Size:           {size_kb} kB\n"
        f"Rss:            {rss} kB\n"
        f"Pss:            {pss} kB\n"
        f"Shared_Clean:          0 kB\n"
        f"Private_Dirty:  {rss} kB\n"
    )


#: The PP0 rank as recorded: two 16 GiB pinned weight images, one 512 MiB
#: draft image, NCCL transport segments, and driver maps.
RECORDED_PP0 = (
    _vma(0x7CCA38000000, 16 * 1024 * 1024, "rw-s", "/dev/zero (deleted)")
    + _vma(0x7CD238000000, 16 * 1024 * 1024, "rw-s", "/dev/zero (deleted)")
    + _vma(0x7CD638000000, 512 * 1024, "rw-s", "/dev/zero (deleted)")
    + _vma(0x7CD0BA000000, 8 * 1024, "rw-s", "/dev/zero (deleted)")
    + _vma(0x7CD0BB000000, 64, "rw-s", "/dev/shm/nccl-FFd7Rq (deleted)")
    + _vma(0x7CD0BC000000, 128, "rw-s", "/dev/shm/psm_b571a8c9")
    + _vma(0x7CD0BD000000, 2048, "rw-s", "/dev/nvidiactl")
    # A PRIVATE mapping of the same device: must not be counted at all.
    + _vma(0x7CD0BE000000, 4096, "rw-p", "/dev/zero (deleted)")
    # A shared file-backed mapping: its own class.
    + _vma(0x7CD0BF000000, 256, "rw-s", "/var/lib/something.bin")
)


class Classification(unittest.TestCase):
    def test_anon_shared_is_recognised_by_the_shared_bit(self):
        self.assertEqual(
            classify_mapping("/dev/zero (deleted)", "rw-s"), CLASS_ANON_SHARED
        )
        self.assertEqual(classify_mapping("", "rw-s"), CLASS_ANON_SHARED)

    def test_a_private_mapping_is_not_a_shared_mapping(self):
        """The bit, not the name. A private /dev/zero map is ordinary anon."""
        self.assertIsNone(classify_mapping("/dev/zero (deleted)", "rw-p"))
        self.assertIsNone(classify_mapping("/dev/nvidiactl", "rw-p"))

    def test_nccl_is_split_from_our_own_named_segments(self):
        self.assertEqual(
            classify_mapping("/dev/shm/nccl-FFd7Rq (deleted)", "rw-s"), CLASS_SHM_NCCL
        )
        self.assertEqual(
            classify_mapping("/dev/shm/psm_b571a8c9", "rw-s"), CLASS_SHM_NAMED
        )

    def test_driver_and_file_classes(self):
        self.assertEqual(classify_mapping("/dev/nvidiactl", "rw-s"), CLASS_DRIVER)
        self.assertEqual(classify_mapping("/dev/nvidia-uvm", "rw-s"), CLASS_DRIVER)
        self.assertEqual(classify_mapping("/var/lib/x.bin", "rw-s"), CLASS_FILE_SHARED)


class ParseRecordedRank(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(prefix="smaps695_")
        with os.fdopen(fd, "w") as handle:
            handle.write(RECORDED_PP0)
        self.mappings = parse_shared_mappings(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_the_private_mapping_is_excluded(self):
        self.assertEqual(len(self.mappings), 8)
        self.assertTrue(all(m.owner_class is not None for m in self.mappings))

    def test_anon_shared_is_the_dominant_class(self):
        anon = [m for m in self.mappings if m.owner_class == CLASS_ANON_SHARED]
        total = sum(m.pss_bytes for m in anon)
        # 16 + 16 GiB images + 512 MiB draft + 8 MiB workspace
        self.assertEqual(len(anon), 4)
        self.assertEqual(total, 32 * GIB + 512 * MIB + 8 * MIB)

    def test_the_two_weight_images_are_flagged_as_big(self):
        census = HostShmemCensus(pid=1)
        for m in self.mappings:
            census.by_class_pss[m.owner_class] = (
                census.by_class_pss.get(m.owner_class, 0) + m.pss_bytes
            )
            census.by_class_count[m.owner_class] = (
                census.by_class_count.get(m.owner_class, 0) + 1
            )
        self.assertEqual(census.by_class_pss[CLASS_ANON_SHARED], 32 * GIB + 520 * MIB)

    def test_a_missing_smaps_is_empty_not_an_exception(self):
        self.assertEqual(parse_shared_mappings("/proc/0/does-not-exist"), [])


class Totals(unittest.TestCase):
    def _census(self):
        c = HostShmemCensus(pid=1)
        c.by_class_pss = {
            CLASS_ANON_SHARED: 32 * GIB,
            CLASS_SHM_NCCL: 256 * MIB,
            CLASS_DRIVER: 4 * GIB,
        }
        c.by_class_count = {CLASS_ANON_SHARED: 3, CLASS_SHM_NCCL: 30, CLASS_DRIVER: 32}
        return c

    def test_driver_mappings_are_measured_but_not_charged(self):
        c = self._census()
        self.assertEqual(c.host_ram_pss, 32 * GIB + 256 * MIB)
        # ...and it is still reported, so a reader can see it was considered.
        self.assertIn("driver", render_host_shmem_line(c))
        self.assertIn("not host RAM", render_host_shmem_line(c))

    def test_the_total_is_pss_not_rss(self):
        """Pss is what reconciles against the cgroup; Rss double-counts.

        Two ranks sharing one 4 GiB segment each report Rss 4 GiB but Pss
        2 GiB, and the cgroup is charged 4 GiB once. Summing Rss over the
        ranks would claim 8 GiB.
        """
        text = _vma(
            0x1000000,
            4 * 1024 * 1024,
            "rw-s",
            "/dev/zero (deleted)",
            pss_kb=2 * 1024 * 1024,
            rss_kb=4 * 1024 * 1024,
        )
        fd, path = tempfile.mkstemp(prefix="smaps695pss_")
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        try:
            mappings = parse_shared_mappings(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0].pss_bytes, 2 * GIB)
        self.assertEqual(mappings[0].rss_bytes, 4 * GIB)

    def test_residual_is_measured_minus_declared(self):
        c = self._census()
        c.declared_bytes = 30 * GIB
        self.assertEqual(c.residual_bytes, 2 * GIB + 256 * MIB)

    def test_the_line_is_one_line_and_carries_the_prefix(self):
        c = self._census()
        c.cgroup_shmem = 75 * GIB
        c.oom_kills = 9
        line = render_host_shmem_line(c, rank=2)
        self.assertNotIn("\n", line)
        self.assertTrue(line.startswith("HOST-SHMEM rank2"))
        self.assertIn("oom_kills=9", line)
        self.assertIn("cgroup_shmem=75.00GiB", line)

    def test_an_unset_ceiling_says_so_rather_than_printing_a_number(self):
        """`memory.max` reads `max` on this rig; the line must not fake one."""
        c = self._census()
        c.cgroup_max = None
        self.assertIn("cgroup_max=unset", render_host_shmem_line(c))


class LiveCollection(unittest.TestCase):
    """The collector must work against this very process, not just fixtures.

    Desk-written code that only ever meets a fixture is unvalidated (#605):
    the /proc walk runs here for real.
    """

    def test_collect_runs_and_is_self_consistent(self):
        from sglang.srt.mem_ledger.host_shmem import collect_host_shmem_census

        c = collect_host_shmem_census()
        self.assertEqual(c.pid, os.getpid())
        self.assertGreaterEqual(c.host_ram_pss, 0)
        for name, value in c.by_class_pss.items():
            self.assertGreaterEqual(value, 0, name)
        self.assertNotIn("\n", render_host_shmem_line(c))

    def test_a_new_anon_shared_mapping_shows_up(self):
        """The measurement moves when the thing it measures moves."""
        import mmap

        from sglang.srt.mem_ledger.host_shmem import collect_host_shmem_census

        before = collect_host_shmem_census().by_class_pss.get(CLASS_ANON_SHARED, 0)
        size = 64 * MIB
        region = mmap.mmap(
            -1,
            size,
            flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            region.write(b"\x01" * size)  # fault every page in
            after = collect_host_shmem_census().by_class_pss.get(CLASS_ANON_SHARED, 0)
            self.assertGreaterEqual(after - before, size - 4096)
        finally:
            region.close()


if __name__ == "__main__":
    unittest.main()
