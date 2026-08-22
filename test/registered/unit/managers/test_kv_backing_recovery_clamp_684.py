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

THE DEFECT WAS A MISSING CLAMP. ``recover`` bounded its target two ways -- by
``_rows_at_boot`` and by what the free column can afford above the corridor law
-- and by nothing else. The pool's reservation was never consulted.

WHY THE REMEMBERED NUMBER COULD EXCEED IT, ROOT-CAUSED. The reservation is
IMMUTABLE: ``KvVmmBufferOwner`` takes ``reserved_num_tokens=self.size`` at
construction (memory_pool.py:2458) and assigns ``_reserved_num_tokens`` exactly
once (kv_vmm_backing.py:979). ``size`` is NOT immutable -- the #330 dial writes
it on every step, as #662-F4 already noted one layer up. So a target derived
from a remembered or configured row count can sit above a ceiling that never
moves, and ``_check_final`` then refuses it identically, forever. The candidate
this task was opened on -- the flip's "released 1410.0 MiB of weights-arena
tail" -- is REFUTED twice over: it is the weights arena, not the KV pool, and
an immutable reservation cannot be what moved.

THE REPAIR IS TO ASK THE BOUND RATHER THAN TRUST THE DERIVATION, the same
correction as #681 (count vs leaf frontier) and #682 (guard ceiling vs the
bound the scheduler holds). ``recover`` now clamps to the arena's ceiling AND
corrects the remembered number, because a clamp alone would only convert a
loud failure into a quiet one that re-clamps to the same place forever.

WHAT THIS FILE IS. A hermetic reproduction, no GPU: the pool is a fake whose
``runtime_set_backing_rows`` refuses exactly as the production one does.
``test_recovery_is_refused_forever_because_nothing_clamps_it`` keeps its name
because it is the acceptance pin the fix was required to invert -- it asserted
the defect before the clamp and asserts the repair after it.

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
        #: What the clamp reads: the arena's IMMUTABLE reservation, pinned to
        #: the pool's size at construction (kv_vmm_backing.py:2458/979).
        self.reserved_backing_rows = int(reserved_rows)
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
        """THE ACCEPTANCE PIN, INVERTED BY THE FIX (#684).

        It used to assert the defect: target 270646 against reservation
        190596, refused, 0 returned, 59 times. It now asserts the repair --
        the target is clamped to what the pool can actually accept, and the
        recovery that was impossible becomes the largest one that is possible.

        Production numbers throughout: boot row count 270646, reservation
        190596, and a free column with room to spare so affordability is not
        what bounds the target.
        """
        pool = _FakeVmmPool(backed_rows=110592, reserved_rows=190596)
        relief = _relief(pool, free_mib=8192, rows_at_boot=270646)

        self.assertGreater(
            relief._rows_at_boot,
            pool.reserved,
            "test setup: the boot row count must exceed the reservation",
        )
        relief.recover()

        self.assertEqual(
            190596,
            pool.full_pool_backed_rows,
            "recovery must reach the reservation, not stop at the old level",
        )
        self.assertTrue(pool.attempts, "recovery must at least have tried")
        self.assertTrue(
            all(a <= pool.reserved for a in pool.attempts),
            f"no attempt may aim above the reservation: {pool.attempts}",
        )

    def test_the_stale_boot_row_count_is_corrected_not_retried(self):
        """59 of 59 in production: no attempt taught the next one anything.

        Clamping alone would stop the exception and keep aiming at 270646 for
        the rest of the process. The remembered number is now corrected to
        what the pool proved it can hold, so the second call has nothing left
        to do instead of failing again.
        """
        pool = _FakeVmmPool(backed_rows=110592, reserved_rows=190596)
        relief = _relief(pool, free_mib=8192, rows_at_boot=270646)
        relief.recover()
        # Either cleared (fully recovered, nothing left to remember) or at
        # most the ceiling. What must NEVER survive is the impossible number.
        self.assertTrue(
            relief._rows_at_boot is None or relief._rows_at_boot <= pool.reserved,
            f"the unsatisfiable target survived the call that proved it: "
            f"{relief._rows_at_boot}",
        )
        before = len(pool.attempts)
        for _ in range(4):
            relief.recover()
        self.assertEqual(
            before,
            len(pool.attempts),
            "a pool already at its reservation must not be asked again",
        )

    def test_a_pool_without_a_reservation_keeps_its_previous_behaviour(self):
        """The clamp must not change pools that expose no arena at all.

        A non-VMM pool has no ceiling to read, so the guard stays inert and
        the path behaves exactly as it did before #684 -- which for this
        (impossible) target still means a refusal.
        """
        pool = _FakeVmmPool(backed_rows=110592, reserved_rows=190596)
        del pool.reserved_backing_rows
        relief = _relief(pool, free_mib=8192, rows_at_boot=270646)
        relief.recover()
        self.assertEqual([270646], pool.attempts, "the target must be unclamped")
        self.assertEqual(110592, pool.full_pool_backed_rows)

    def test_a_zero_reservation_reads_as_absent_not_as_a_ceiling_of_zero(self):
        """0 means 'no arena', never 'a reservation of zero'.

        Clamping to 0 would ask the pool to unback itself completely from
        inside recovery -- a shrink wearing a grow's name.
        """
        pool = _FakeVmmPool(backed_rows=110592, reserved_rows=190596)
        pool.reserved_backing_rows = 0
        relief = _relief(pool, free_mib=8192, rows_at_boot=270646)
        relief.recover()
        self.assertTrue(
            all(a > 0 for a in pool.attempts),
            f"a zero ceiling must never become the target: {pool.attempts}",
        )

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

    def test_the_clamp_can_only_lower_the_target_never_raise_it(self):
        """The safety property that makes this shippable without a window.

        The clamp runs AFTER the corridor-affordability bound, so when both
        bite the target is the smaller of the two. If it could raise the
        target it would commit pages the corridor law had already refused --
        the failure that drove rank 1 to 6 MiB free and OOMed inside relief.

        Affordability here allows 39408 rows above the current 110592; the
        reservation would allow 190596. The target must be the former.
        """
        pool = _FakeVmmPool(backed_rows=110592, reserved_rows=190596)
        affordable_rows = 39408
        free_mib = LAW_FLOOR // (1 << 20) + (affordable_rows * BYTES_PER_ROW) // (
            1 << 20
        )
        relief = _relief(pool, free_mib=free_mib, rows_at_boot=270646)
        relief.recover()
        self.assertTrue(pool.attempts, "recovery must have been attempted")
        self.assertLessEqual(
            max(pool.attempts),
            110592 + affordable_rows,
            f"the clamp raised the target past what the corridor allowed: "
            f"{pool.attempts}",
        )
        self.assertLess(
            max(pool.attempts), pool.reserved, "affordability must be the binding bound"
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
