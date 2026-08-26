"""#913: the seam copied rows whose pages a backing-dial shrink had released.

THE SPECIMEN. R7 of the 0826 acceptance window
(``/spinning/evidence-665-f1/boot_accept0826r7fix_0826_1817.log``, 18:27:14Z)
died with a CUDA illegal memory access on this exact chain::

    _release_residents_for_cutover -> release_residents_for_cutover
      -> _retract -> retract_all -> release_req -> seam_copy_state
      -> Req.offload_kv_cache -> paged.get_cpu_copy
      -> MHATokenToKVPool.get_cpu_copy -> current_platform.synchronize()

The traceback names ``synchronize()``, which is the aftermath: the fault was
raised by the indexed read one frame earlier and surfaced at the next sync.
The same boot logged the cause on its own line, on all three ranks and on
every one of the 93 flips::

    KV-OWNERSHIP VIOLATION (pre-cutover tp_to_pp) [coverage] 466934 rows:
      ... claimed row id(s) sit at or above the committed backing ...
      these are live rows that are already unmapped

and, for the rank that died, the arithmetic behind it::

    PP2  KV-BACKING UNDER-BACKED RANK: floor 126995 rows exceeds the 114688
         rows this rank has backed ... resident/parked ceiling 122898 is at
         or above the high-water row 122898

122898 > 114688. PP2 held live row ids ~8200 above its own committed backing.

WHY NOTHING STOPPED IT, which is the part worth pinning.

  * ``_enforce_exposure_at_seam`` (#851) DID run, on that flip, and reported
    "the exposed id space is within its backing, nothing withdrawn" -- and it
    was right. It enforces LAW 1 (exposure <= committed) by lowering what the
    allocator may hand out NEXT. Rows already handed out are LAW 2 (coverage),
    and Law 2 has no enforcer: the audit that computes it is wrapped in
    ``except Exception: # noqa: BLE001 -- an instrument, never a gate``.
  * ``check_cpu_copy_rows`` (#783b) DID run, and passed, because its bound was
    ``k_buffer[0].shape[0]`` -- the pool's IMMUTABLE BOOT VA RESERVATION, which
    ``store_bound_rows`` documents as exactly that. Physical backing is
    ``uniform_backed_rows`` and moves on every dial shrink. The guard checked
    the axis that cannot change against a hazard that lives on the one that
    does. Same shape as the #718 guard that could not see #905 ("49 < 30518"):
    a range check against the wrong number.

TWO DEFECTS, AND THIS FILE COVERS THE SECOND.

  ROOT-A (upstream, NOT fixed here): ``runtime_set_backing_tokens`` decommits
    on the premise, stated in its own comment, that "rows above ``n`` are dead
    the moment ``size`` is ``n``". That is false while a resident holds one.
    Filed; the remedy is on the FLOOR side and cannot be taken at the actuator
    without letting one rank's floor freeze the group's shrink, which the same
    boot shows happening ("DECLINED by a PEER FLOOR").
  ROOT-B (here): the copy had no mapped-proof and so converted ROOT-A into an
    unrecoverable rank death instead of a named refusal. That is its own root,
    not ROOT-A's effect: the bound was on the wrong axis independently of
    whether the dial ever misbehaves.

CONSERVATIVE BY CHOICE. Refusing at the copy risks dropping a prefix that was
in fact readable -- cost: a recompute, bounded, never a wrong answer. Making
the unmapper wait for the seam risks a rank under no pressure vetoing a shrink
a pressed rank needs -- cost: the #848 latch, an unbounded flip stall. The
cheaper failure direction is the one built.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import CpuCopyUnmappedRows, check_cpu_copy_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# The measured R7 numbers, used as-is so the test names the specimen.
RESERVATION_ROWS = 485366  # reserved_backing_rows, the immutable VA ceiling
BACKED_ROWS = 114688  # uniform_backed_rows after the dial shrink
LIVE_HIGH_WATER = 122898  # highest row id PP2's live set still held


def rows(*ids):
    return torch.tensor(list(ids), dtype=torch.int64)


class TestTheGuardOnTheOldAxisCannotSeeIt(CustomTestCase):
    """Pins WHY the shipped guard passed, so the fix is not mistaken for a
    tightening of something that already worked."""

    def test_the_fatal_row_is_inside_the_reservation(self):
        self.assertLess(LIVE_HIGH_WATER, RESERVATION_ROWS)

    def test_reservation_axis_alone_accepts_the_fatal_row(self):
        # No backed_rows: byte-identical to the pre-#913 call. It passes.
        check_cpu_copy_rows(rows(LIVE_HIGH_WATER), RESERVATION_ROWS, "offload", "row")


class TestTheCopyRefusesAnUnmappedRow(CustomTestCase):
    def test_a_row_above_the_backing_is_refused(self):
        with self.assertRaises(CpuCopyUnmappedRows) as cm:
            check_cpu_copy_rows(
                rows(LIVE_HIGH_WATER),
                RESERVATION_ROWS,
                "offload",
                "row",
                backed_rows=BACKED_ROWS,
            )
        msg = str(cm.exception)
        self.assertIn(str(LIVE_HIGH_WATER), msg)
        self.assertIn(str(BACKED_ROWS), msg)

    def test_the_whole_extent_is_refused_when_only_its_tail_is_unmapped(self):
        """NOT A CLAMP. The mapped rows are a prefix of the extent only by
        accident; copying them under the request's full length writes someone
        else's KV at restore -- a wrong answer, which is worse than slow."""
        with self.assertRaises(CpuCopyUnmappedRows):
            check_cpu_copy_rows(
                rows(10, 20, BACKED_ROWS - 1, BACKED_ROWS),
                RESERVATION_ROWS,
                "offload",
                "row",
                backed_rows=BACKED_ROWS,
            )

    def test_rows_within_the_backing_still_pass(self):
        check_cpu_copy_rows(
            rows(1, 2, BACKED_ROWS - 1),
            RESERVATION_ROWS,
            "offload",
            "row",
            backed_rows=BACKED_ROWS,
        )

    def test_the_restore_direction_is_guarded_too(self):
        """`load_cpu_copy` WRITES those rows. Guarding only the read would
        leave the write half addressing released pages."""
        with self.assertRaises(CpuCopyUnmappedRows):
            check_cpu_copy_rows(
                rows(LIVE_HIGH_WATER),
                RESERVATION_ROWS,
                "restore",
                "row",
                backed_rows=BACKED_ROWS,
            )


class TestTheTwoRefusalsStayDistinguishable(CustomTestCase):
    """The seam swallows one of these and must never swallow the other."""

    def test_an_id_outside_the_reservation_is_the_older_defect(self):
        with self.assertRaises(ValueError) as cm:
            check_cpu_copy_rows(
                rows(RESERVATION_ROWS + 1),
                RESERVATION_ROWS,
                "offload",
                "row",
                backed_rows=BACKED_ROWS,
            )
        self.assertNotIsInstance(cm.exception, CpuCopyUnmappedRows)

    def test_a_negative_id_is_the_older_defect(self):
        with self.assertRaises(ValueError) as cm:
            check_cpu_copy_rows(
                rows(-1), RESERVATION_ROWS, "offload", "row", backed_rows=BACKED_ROWS
            )
        self.assertNotIsInstance(cm.exception, CpuCopyUnmappedRows)

    def test_the_unmapped_refusal_is_a_valueerror_subclass(self):
        """Callers that only knew the old contract keep working."""
        self.assertTrue(issubclass(CpuCopyUnmappedRows, ValueError))


class TestAPoolWithNoArenaIsUnaffected(CustomTestCase):
    """None is 'cannot answer', never 'nothing is backed'.

    ``uniform_backed_rows`` returns 0 on an eager pool. Read as a bound, 0
    would refuse every copy off the dial lane -- the #832 rule ("a claim over
    an empty set is not the absence of a claim") in its second instance.
    """

    def test_none_disables_the_backing_check_entirely(self):
        check_cpu_copy_rows(
            rows(LIVE_HIGH_WATER),
            RESERVATION_ROWS,
            "offload",
            "row",
            backed_rows=None,
        )

    def test_empty_and_absent_indices_are_still_no_ops(self):
        check_cpu_copy_rows(None, RESERVATION_ROWS, "offload", "row", backed_rows=1)
        check_cpu_copy_rows(
            torch.tensor([], dtype=torch.int64),
            RESERVATION_ROWS,
            "offload",
            "row",
            backed_rows=1,
        )


class TestTheSeamDeclinesInsteadOfDying(CustomTestCase):
    """The half that keeps the rank alive.

    A refusal that propagates out of `seam_copy_state` still kills the flip:
    `release_residents_for_cutover` is past the no-return point and its caller
    has no abort left. The response has to be the DECLINE path that already
    exists for a mid-chunk request.
    """

    def test_seam_copy_state_catches_only_the_unmapped_refusal(self):
        import inspect

        from sglang.srt.managers import schedule_batch

        src = inspect.getsource(schedule_batch.seam_copy_state)
        self.assertIn("except CpuCopyUnmappedRows", src)
        # Narrow: a bare `except ValueError` would swallow the #783b defect.
        self.assertNotIn("except ValueError", src)
        self.assertIn("declined_unmapped", src)

    def test_the_decline_is_counted_apart_from_the_mid_chunk_decline(self):
        from sglang.srt.managers.schedule_batch import _SEAM_STATE_COUNTS

        self.assertIn("declined_unmapped", _SEAM_STATE_COUNTS)
        self.assertIn("declined", _SEAM_STATE_COUNTS)

    def test_a_declined_copy_leaves_no_half_taken_state(self):
        """A partially populated copy would be applied at restore against an
        extent it does not cover -- the W38-A IndexError, by another route."""
        import inspect

        from sglang.srt.managers import schedule_batch

        src = inspect.getsource(schedule_batch.seam_copy_state)
        for field in (
            "req.kv_cache_cpu = None",
            "req.kv_cache_cpu_extent = None",
            "req.kv_cache_cpu_layout = None",
        ):
            self.assertIn(field, src)


class TestTheCopySitesActuallyPassTheBackingAxis(CustomTestCase):
    """PRESENT-AND-VERDRAHTET. A guard that cannot fire is the shape this
    ticket exists to remove, and it is the middle of the three delivery states
    that is most expensive to mistake for either end."""

    def test_both_mha_copy_directions_pass_a_backing_bound(self):
        import inspect

        from sglang.srt.mem_cache import memory_pool

        for name in ("get_cpu_copy", "load_cpu_copy"):
            src = inspect.getsource(getattr(memory_pool.MHATokenToKVPool, name))
            self.assertIn(
                "backed_rows=self._committed_row_bound()",
                src,
                f"MHATokenToKVPool.{name} still bounds only on the reservation",
            )

    def test_the_backing_bound_reads_the_arena_not_the_tensor(self):
        import inspect

        from sglang.srt.mem_cache import memory_pool

        src = inspect.getsource(memory_pool.MHATokenToKVPool._committed_row_bound)
        self.assertIn("uniform_backed_tokens", src)
        # None, not 0, when there is no arena.
        self.assertIn("return None", src)


if __name__ == "__main__":
    unittest.main()
