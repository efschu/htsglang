"""#695: shmem is charged to ``file``, and with no swap it is not reclaimable.

THE DEFECT, in one sentence
---------------------------
``honest_host_memory_bytes`` subtracts ``anon + kernel`` from the ceiling and
deliberately does not subtract ``file``, on the stated grounds that page cache
is reclaimable. In cgroup v2, tmpfs/shmem pages are accounted in ``file`` and
NOT in ``anon`` -- so a process holding tens of GiB of ``MAP_SHARED`` memory is
invisible to every guard that consults this function, and the guard reports
that memory as free.

WHAT IT COST
------------
Measured on the live PP=3 boot, 2026-08-12, ``/sys/fs/cgroup/.lxc``::

    anon           14596120576   (13.6 GiB)
    file           91118194688   (84.9 GiB)
    shmem          80607027200   (75.07 GiB)   <-- inside `file`, not `anon`
    file_mapped    81086967808
    memory.current 106357735424  (99.06 GiB)
    memory.events  oom_kill 9

``/proc/meminfo`` on the same boot: ``SwapTotal: 0``. Shmem with no swap has
nowhere to be reclaimed TO: the pages are pinned for the lifetime of the
mapping. The ledger nevertheless counted all 75.07 GiB as reclaimable cache.
Nine cumulative cgroup OOM kills, including the SIGKILL that presented as a
silent rank death.

WHY THE ORIGINAL REASONING WAS RIGHT AND STILL PRODUCED THIS
------------------------------------------------------------
"page cache is reclaimable, do not subtract it" is correct for page cache. The
error is that ``file`` is not only page cache. The fix is therefore not to
start subtracting ``file`` -- that would refuse boots that would have
succeeded, exactly as the original docstring warns -- but to subtract the
UNRECLAIMABLE SUBSET of it, which the cgroup already reports under its own key.

Reclaimability of shmem is a function of swap, so the rule is stated against
swap rather than as a constant: shmem can be pushed to swap, and only the part
that exceeds free swap is unreclaimable. With ``SwapTotal: 0`` that reduces to
"all of it", which is this rig.

CAN-FAIL
--------
Delete the ``shmem`` subtraction in ``honest_host_memory_bytes`` and
``test_shmem_without_swap_is_not_available`` goes red. Subtract shmem
unconditionally, ignoring swap, and
``test_shmem_backed_by_free_swap_stays_available`` goes red. Stop reading the
``shmem`` key in ``_read_cgroup_memory`` and
``test_live_cgroup_reader_reports_shmem`` goes red.
"""

import unittest

from sglang.srt.memtier.profile import (
    _read_cgroup_memory,
    honest_host_memory_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GIB = 1024**3


class ShmemIsUnreclaimableWithoutSwap(unittest.TestCase):
    """The live shape: a finite ceiling, and 75 GiB of it spent on shmem."""

    def test_shmem_without_swap_is_not_available(self):
        # The rig's own numbers, rounded to whole GiB: a 120 GiB ceiling with
        # 14 GiB anon, 2 GiB kernel and 75 GiB of shmem, and no swap.
        total, available = honest_host_memory_bytes(
            meminfo_total=120 * GIB,
            meminfo_available=None,
            cgroup_max=120 * GIB,
            cgroup_anon=14 * GIB,
            cgroup_kernel=2 * GIB,
            cgroup_shmem=75 * GIB,
            swap_free=0,
        )
        self.assertEqual(total, 120 * GIB)
        # 120 - 14 - 2 - 75. Before the fix this answered 104 GiB, i.e. it
        # promised 75 GiB that no allocation could ever have.
        self.assertEqual(available, 29 * GIB)

    def test_shmem_backed_by_free_swap_stays_available(self):
        """Only the part of shmem that exceeds free swap is unreclaimable."""
        total, available = honest_host_memory_bytes(
            meminfo_total=120 * GIB,
            meminfo_available=None,
            cgroup_max=120 * GIB,
            cgroup_anon=14 * GIB,
            cgroup_kernel=2 * GIB,
            cgroup_shmem=75 * GIB,
            swap_free=75 * GIB,
        )
        self.assertEqual(total, 120 * GIB)
        # All of shmem could be pushed to swap, so none of it is subtracted.
        self.assertEqual(available, 104 * GIB)

    def test_partial_swap_subtracts_only_the_excess(self):
        total, available = honest_host_memory_bytes(
            meminfo_total=120 * GIB,
            meminfo_available=None,
            cgroup_max=120 * GIB,
            cgroup_anon=14 * GIB,
            cgroup_kernel=2 * GIB,
            cgroup_shmem=75 * GIB,
            swap_free=25 * GIB,
        )
        self.assertEqual(available, 54 * GIB)  # 104 - (75 - 25)

    def test_unknown_shmem_degrades_to_the_old_answer(self):
        """A reader that cannot establish shmem must not invent one."""
        total, available = honest_host_memory_bytes(
            meminfo_total=120 * GIB,
            meminfo_available=None,
            cgroup_max=120 * GIB,
            cgroup_anon=14 * GIB,
            cgroup_kernel=2 * GIB,
            cgroup_shmem=None,
            swap_free=None,
        )
        self.assertEqual(available, 104 * GIB)

    def test_shmem_is_not_double_counted_against_anon(self):
        """shmem lives in `file`; subtracting it must not also hit `anon`.

        Pins the containment relation the fix depends on. If a future kernel
        or a future edit moved shmem into ``anon``, this arithmetic would
        double-count it, and the assertion below is what would catch it: the
        answer for "all memory is shmem, nothing anonymous" must be the
        ceiling minus shmem exactly once.
        """
        _, available = honest_host_memory_bytes(
            meminfo_total=100 * GIB,
            meminfo_available=None,
            cgroup_max=100 * GIB,
            cgroup_anon=0,
            cgroup_kernel=0,
            cgroup_shmem=40 * GIB,
            swap_free=0,
        )
        self.assertEqual(available, 60 * GIB)

    def test_the_no_ceiling_branch_also_prices_shmem(self):
        """The live rig reads ``memory.max = max``; that branch needs it too.

        With no finite ceiling the function clamps by ``MemAvailable``. On a
        container whose ``/proc/meminfo`` is synthesised, that clamp is exactly
        the number the module docstring says cannot be trusted, so the shmem
        subtraction has to apply here as well rather than only on the
        finite-``memory.max`` branch.
        """
        _, available = honest_host_memory_bytes(
            meminfo_total=120 * GIB,
            meminfo_available=110 * GIB,
            cgroup_max=None,
            cgroup_anon=14 * GIB,
            cgroup_kernel=2 * GIB,
            cgroup_shmem=75 * GIB,
            swap_free=0,
        )
        self.assertEqual(available, 29 * GIB)

    def test_available_never_goes_negative(self):
        _, available = honest_host_memory_bytes(
            meminfo_total=32 * GIB,
            meminfo_available=None,
            cgroup_max=32 * GIB,
            cgroup_anon=20 * GIB,
            cgroup_kernel=4 * GIB,
            cgroup_shmem=30 * GIB,
            swap_free=0,
        )
        self.assertEqual(available, 0)


class LiveCgroupReader(unittest.TestCase):
    """The reader must actually surface the new keys, not just accept them.

    A pure function nobody feeds is a fix that does not ship (#605): the
    arithmetic above is only reached if ``_read_cgroup_memory`` returns shmem.
    """

    def test_live_cgroup_reader_reports_shmem(self):
        reading = _read_cgroup_memory()
        self.assertEqual(
            len(reading),
            5,
            "expected (max, anon, kernel, shmem, swap_free) from the reader",
        )
        limit, anon, kernel, shmem, swap_free = reading
        for value in (anon, kernel, shmem, swap_free):
            if value is not None:
                self.assertGreaterEqual(value, 0)

    def test_reader_survives_a_missing_cgroupfs(self):
        """cgroup v1 / no cgroupfs: all-None, never an exception."""
        import sglang.srt.memtier.profile as profile_mod

        real_open = open

        def _refuse(path, *args, **kwargs):
            if str(path).startswith("/sys/fs/cgroup"):
                raise OSError("no cgroupfs")
            return real_open(path, *args, **kwargs)

        original = profile_mod.open if hasattr(profile_mod, "open") else None
        profile_mod.open = _refuse  # type: ignore[attr-defined]
        try:
            limit, anon, kernel, shmem, swap_free = profile_mod._read_cgroup_memory()
        finally:
            if original is None:
                del profile_mod.open  # type: ignore[attr-defined]
            else:
                profile_mod.open = original  # type: ignore[attr-defined]
        self.assertIsNone(limit)
        self.assertIsNone(anon)
        self.assertIsNone(kernel)
        self.assertIsNone(shmem)


if __name__ == "__main__":
    unittest.main()
