"""#684: recovery aims at a row count the pool cannot accept, so the cap sticks.

MEASURED, AND UNCONFOUNDED. On 2026-08-16, from 02:15:24 to 02:35:26, this line
appeared 59 times at a steady 3 per minute -- once per rank, once per flip leg:

    KV-BACKING recovery to 270646 rows failed: final_num_tokens=270646 must
    satisfy page_size=1 <= final <= reserved=190596. The cap stays engaged, so
    admission capacity remains reduced -- a capacity loss, never a fault.

59 attempts, 59 failures, target always ABOVE the pool's reservation
(270646/190596, 180428/108912, 179466/136140). The window starts before any
test-harness CUDA activity on that rig, so unlike the free-column readings from
02:29 onward these lines are not confounded by a competing process.

WHY IT MATTERS MORE THAN THE LINE ITSELF. Recovery is what LIFTS the backing
cap. While it fails the cap stays engaged, so the pool stays shrunk, so every
later ``free_up_to`` finds the backing already at or below its target and
honestly claims 0 MiB -- and reports that through a message asserting "this
pool cannot pay: the arena has no commit chunk", which is a mechanism the
surrounding code itself knows may not apply. That is how the corridor guard's
only escalation rung above ``allocator-cache`` stayed dead for the whole boot
while its diagnostic pointed at the arena.

THE DEFECT IS A MISSING CLAMP. ``recover`` bounds its target two ways -- by
``_rows_at_boot`` and by what the free column can afford above the corridor law
-- and by nothing else. The pool's own reservation is never consulted; the
string ``reserved`` does not appear against a pool anywhere in the module. So
when the reservation is smaller than the boot row count the target is
unsatisfiable BY CONSTRUCTION and every attempt fails identically, forever.

WHAT THIS FILE IS. A hermetic reproduction, no GPU: the pool is a fake whose
``runtime_set_backing_rows`` refuses exactly as the production one does. It
pins the defect as it stands today. When the clamp lands,
``test_recovery_is_refused_forever_because_nothing_clamps_it`` INVERTS -- the
recovery returns bytes and the cap lifts -- and that inversion is the fix's
acceptance test, already written.

NOT WHAT THE TASK WAS ORIGINALLY FRAMED AS. The suspected mechanism was extent
fragmentation ("191k rows of slack, zero releasable extents"). It is not:
``KvBackingRelief`` releases by lowering backing "to just above the highest
live row", a high-water-mark tail policy, and the log's own
``highest live row`` field corroborates it (at row 122 a shrink released 1322
MiB; at row 234118 it released nothing worth having). Fragmentation was the
wrong frame; the unliftable cap is the defect.
"""

import unittest
from typing import Optional

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

BYTES_PER_ROW = 4096
LAW_FLOOR = 1024 * 1024 * 1024


class _FakeVmmPool:
    """A VMM-backed pool that refuses a target above its reservation.

    The refusal text is the production one, verbatim, so a reader can match
    this test to the 02:15-02:35 log lines without translating.
    """

    def __init__(self, backed_rows: int, reserved_rows: int, page_size: int = 1):
        self.full_pool_backed_rows = int(backed_rows)
        self.reserved = int(reserved_rows)
        self.page_size = int(page_size)
        self.attempts = []

    def runtime_set_backing_rows(self, rows: int) -> None:
        self.attempts.append(int(rows))
        if not (self.page_size <= int(rows) <= self.reserved):
            raise ValueError(
                f"final_num_tokens={int(rows)} must satisfy "
                f"page_size={self.page_size} <= final <= reserved={self.reserved}"
            )
        self.full_pool_backed_rows = int(rows)


def _relief(pool: _FakeVmmPool, *, free_mib: int, rows_at_boot: Optional[int]):
    from sglang.srt.managers.kv_backing_relief import KvBackingRelief

    relief = KvBackingRelief(
        pool,
        allocator=None,
        live_slots_fn=lambda: [],
        bytes_per_row=BYTES_PER_ROW,
        probe=lambda: free_mib * (1 << 20),
        device_index=0,
        buffers=1,
        law_floor_bytes=LAW_FLOOR,
        # Never re-resolve the active layout: this test is about one pool.
        pool_fn=lambda: None,
    )
    relief._rows_at_boot = rows_at_boot
    return relief


class TheRecoveryTargetIgnoresTheReservation(unittest.TestCase):
    """The 02:15-02:35 loop, reproduced from injected state alone."""

    def test_recovery_is_refused_forever_because_nothing_clamps_it(self):
        """PINS THE DEFECT. Inverts when #684's clamp lands.

        Production numbers: boot row count 270646, reservation 190596, and a
        free column with room to spare so affordability is not what bounds
        the target.
        """
        pool = _FakeVmmPool(backed_rows=110592, reserved_rows=190596)
        relief = _relief(pool, free_mib=8192, rows_at_boot=270646)

        self.assertGreater(
            relief._rows_at_boot,
            pool.reserved,
            "test setup: the boot row count must exceed the reservation",
        )
        recovered = relief.recover()

        self.assertEqual(0, recovered, "recovery returned bytes it did not get")
        self.assertEqual(
            110592,
            pool.full_pool_backed_rows,
            "a failed grow must leave the watermark exactly where it was",
        )
        self.assertTrue(pool.attempts, "recovery must at least have tried")
        self.assertTrue(
            all(a > pool.reserved for a in pool.attempts),
            f"every attempt aimed above the reservation: {pool.attempts}",
        )

    def test_it_fails_identically_however_many_times_it_runs(self):
        """59 of 59 in production: no attempt teaches the next one anything."""
        pool = _FakeVmmPool(backed_rows=110592, reserved_rows=190596)
        relief = _relief(pool, free_mib=8192, rows_at_boot=270646)
        for _ in range(5):
            self.assertEqual(0, relief.recover())
        self.assertEqual(5, len(pool.attempts))
        self.assertEqual({270646}, set(pool.attempts), "the target never moves")

    def test_a_reservation_above_the_boot_rows_recovers_normally(self):
        """The control: the same code path works when the clamp is not needed.

        Without this the test above would be satisfied by a recovery that is
        broken for every input rather than for this one.
        """
        pool = _FakeVmmPool(backed_rows=110592, reserved_rows=400000)
        relief = _relief(pool, free_mib=8192, rows_at_boot=270646)
        relief.recover()
        self.assertEqual(
            270646,
            pool.full_pool_backed_rows,
            "with room in the reservation the pool must reach the boot rows",
        )

    def test_affordability_still_bounds_the_target(self):
        """The bound that IS implemented must survive the one that is not.

        A free column with nothing above the corridor law must defer rather
        than aim high -- the pool is not even asked.
        """
        pool = _FakeVmmPool(backed_rows=110592, reserved_rows=400000)
        relief = _relief(pool, free_mib=LAW_FLOOR // (1 << 20), rows_at_boot=270646)
        self.assertEqual(0, relief.recover())
        self.assertEqual([], pool.attempts, "an unaffordable grow must not be tried")


if __name__ == "__main__":
    unittest.main()
