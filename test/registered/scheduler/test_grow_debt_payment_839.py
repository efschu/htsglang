"""#839 B -- a deferred grow is PAID, or refused by name. Never accumulated.

THE SPECIMEN, boot_window4B_0823_2116, /spinning/evidence-665-f1/
window4B_grow_debt_pool_too_small/::

    [#834] GROW DEFERRED:     3
    [#834] GROW PAID:         2
    GROW-DEBT-UNPAID:       588 lines
      294 x  6144 row(s) BACKED but not EXPOSED for 32 rounds (agreed 118784)
      294 x 83968 row(s) BACKED but not EXPOSED for 32 rounds (agreed 118784)
    FLIP ABANDONED (pool too small for the live set):  207
    corridor gate refused the seam staging: want 3114-3124 MiB, free 2788

READ THE FIRST TWO NUMBERS CAREFULLY, because the obvious reading is wrong. The
round hook is NOT failing to run and the grow is NOT failing to happen: two of
three deferrals were paid, and ``GROW PAID`` says so. What never happens is the
SECOND half -- the group agreeing to a level that includes the rows the grow
committed. ``_pay_deferred_grow`` backs the pages and then clamps exposure back
to the level booked when the deferral was made, and #834's own alarm states the
consequence in its own words::

    "the memory is spent and the pool is not getting it, so admission is capped
     against capacity that physically exists. The levelling that would release
     it runs on the seam's collective cadence; if no flip is arming, none will."

THE ROOT IS THAT SENTENCE, and it is a cycle rather than a missing call. The
only actuator that ever RAISED the agreed level was
``level_kv_backing_to_group``, called from ``seam_kv_recover``
(phase_flip_runtime.py:1754) on the ``tp_to_pp`` post-cutover hook. So:

    the level rises only at a cutover
      -> a cutover needs a pool large enough for the live set
        -> the pool's usable size is the agreed level
          -> the level rises only at a cutover

207 abandons is that cycle, once per attempt. It is #814's ratchet with the
teeth on a different gear, which is exactly what #834's guard predicted and
named -- the guard was right and had no creditor to hand the debt to.

WHAT THE FIX IS NOT. It is not "expose the rows when the wait gets
embarrassing": an id one rank exposes and a peer cannot map aborts all three
inside ``store_kvcache``'s bounds assert, which is why
``_deferred_grow_debt_check`` is deliberately an alarm and never an actuator.
And it is not a new collective at a rank-local cadence, which is the
2026-08-08 boots 9/10 wedge shape (HANDOVER-S834).

WHAT IT IS. The rung's own per-round reduction ALREADY carries the group's MIN
backed rows -- ``collective_slot_ballot``'s ``min_backed_rows``, decoded
identically on every rank, on the one path every rank reaches unconditionally
(phase_flip_spill.py:2042). #833 records that number; #839 also PUBLISHES it.
The moment the group's poorest rank has grown too, every rank raises to the new
minimum in the same round, and the debt is settled without a cutover and
without a single new collective.

THE UNPAID COUNTER STAYS. It is the observable that made this findable, and a
fix that removed it would leave the next occurrence silent. What changes is
that it now has a reason to reach zero.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dcp.phase_flip_plan import TP_TO_PP
from sglang.srt.managers import phase_flip_runtime as _rt
from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

BYTES_PER_ROW = 32768
LAW_FLOOR = 1024 * 1024 * 1024

#: The level the group agreed in segment B, on all 36 clamp lines.
W4B_AGREED = 118784
#: The larger of the two standing debts, named 294 times.
W4B_DEBT = 83968
#: So this rank has this much backed and may expose only ``W4B_AGREED``.
W4B_BACKED = W4B_AGREED + W4B_DEBT
#: The id space the allocator spans.
W4B_RESERVATION = 462163


class _FakeAlloc:
    def __init__(self, size: int):
        self.size = int(size)
        self.free_pages = torch.arange(1, int(size) + 1, dtype=torch.int64)
        self.residency_withheld_slots = 0


class _FakeVmmPool:
    def __init__(self, backed_rows: int, reserved_rows: int):
        self.full_pool_backed_rows = int(backed_rows)
        self.reserved = int(reserved_rows)
        self.reserved_backing_rows = int(reserved_rows)
        self.size = int(reserved_rows)
        self.page_size = 1
        self.attempts = []

    def runtime_set_backing_rows(self, rows: int) -> None:
        self.attempts.append(int(rows))
        self.full_pool_backed_rows = int(rows)


def _relief(backed_rows: int = W4B_BACKED):
    from sglang.srt.managers.kv_backing_relief import KvBackingRelief

    pool = _FakeVmmPool(backed_rows, W4B_RESERVATION)
    return KvBackingRelief(
        pool,
        allocator=_FakeAlloc(W4B_RESERVATION),
        live_slots_fn=lambda: torch.empty((0,), dtype=torch.int64),
        bytes_per_row=BYTES_PER_ROW,
        probe=lambda: 8192 * (1 << 20),
        device_index=0,
        buffers=1,
        law_floor_bytes=LAW_FLOOR,
        pool_fn=lambda: None,
    )


def _runtime(relief):
    """A runtime carrying only what the deferred-grow path touches.

    ``__new__`` for the reason test_seam_shrink_834.py states: ``__init__``
    needs a live group and none of these methods do.
    """
    rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    rt._pending = None
    rt._round = 0
    rt._deferred_grow_pending = False
    rt._deferred_grow_rows = 0
    rt._deferred_grow_round = None
    rt._deferred_grow_level = None
    scheduler = SimpleNamespace(tree_cache=SimpleNamespace(cache_controller=None))
    setattr(scheduler, KV_BACKING_RELIEF_ATTR, relief)
    rt._census_scheduler = scheduler
    return rt


def _ballot(relief, floor: int) -> None:
    """One seam round of the rung's reduction, as the rank sees it.

    Both halves, in the order phase_flip_spill.py:2042 runs them: take the
    group's MIN backed rows out of the reduction, then publish this rank's
    exposure at it.
    """
    relief.note_group_backing_floor(floor)
    publish = getattr(relief, "publish_group_exposure", None)
    if publish is not None:
        publish("seam ballot")


class TheDebtIsBookedAndCounted(unittest.TestCase):
    """Before anything is fixed: the specimen's shape must be reachable."""

    def test_a_deferred_grow_leaves_rows_backed_and_unexposed(self):
        relief = _relief()
        rt = _runtime(relief)
        rt.book_deferred_grow(TP_TO_PP, W4B_AGREED)
        rt._round = 1
        rt._pay_deferred_grow()
        self.assertEqual(relief.exposed_rows(), W4B_AGREED)
        self.assertEqual(relief.backed_rows(), W4B_BACKED)
        self.assertEqual(
            rt._deferred_grow_rows,
            W4B_DEBT,
            "the specimen's standing debt, to the row",
        )

    def test_the_unpaid_alarm_names_the_ratchet_and_keeps_counting(self):
        """The observable stays. A fix that silenced it would be worse than
        the defect -- 588 lines is how this was found at all."""
        relief = _relief()
        rt = _runtime(relief)
        rt.book_deferred_grow(TP_TO_PP, W4B_AGREED)
        rt._round = 1
        rt._pay_deferred_grow()
        patience = _rt.seam_shrink_grow_debt_rounds()
        if patience <= 0:
            self.skipTest("debt patience is disabled in this configuration")
        rt._round = 1 + patience
        with self.assertLogs(
            "sglang.srt.managers.phase_flip_runtime", level="ERROR"
        ) as caught:
            rt._deferred_grow_debt_check()
        line = "\n".join(caught.output)
        self.assertIn(_rt.GROW_DEBT_UNPAID, line)
        self.assertIn(str(W4B_DEBT), line)
        self.assertIn("#814", line)


class TheDebtIsPaidWithoutACutover(CustomTestCase):
    """THE FIX. RED ON THE BASE TREE: nothing outside the seam can raise the
    agreed level, so the debt survives every ballot."""

    def test_a_seam_ballot_settles_the_debt(self):
        relief = _relief()
        rt = _runtime(relief)
        rt.book_deferred_grow(TP_TO_PP, W4B_AGREED)
        rt._round = 1
        rt._pay_deferred_grow()
        self.assertEqual(rt._deferred_grow_rows, W4B_DEBT)

        # The group's poorest rank has now grown too, so the next reduction
        # carries a higher minimum. NO CUTOVER HAS HAPPENED and none is armed.
        _ballot(relief, W4B_BACKED)

        rt._round = 2
        rt._pay_deferred_grow()
        self.assertEqual(
            relief.exposed_rows(),
            W4B_BACKED,
            "the group agreed a higher level and this rank did not take it",
        )
        self.assertEqual(
            rt._deferred_grow_rows,
            0,
            "rows stay BACKED but unexposed after the group levelled up to "
            "them -- #814's ratchet, which is what #834 crit 13 measured",
        )

    def test_the_debt_is_settled_within_the_patience_window(self):
        """Paid within N rounds, where N is the deadline the alarm uses.

        N COMES FROM THE DECODE WINDOW, not from this test: the ballot runs on
        every seam round, and ``seam_shrink_grow_debt_rounds`` is the same
        patience #834 already shouts against. Binding the payment to that
        number is what makes "paid or named" a closed statement.
        """
        patience = _rt.seam_shrink_grow_debt_rounds()
        if patience <= 0:
            self.skipTest("debt patience is disabled in this configuration")
        relief = _relief()
        rt = _runtime(relief)
        rt.book_deferred_grow(TP_TO_PP, W4B_AGREED)
        rt._round = 1
        rt._pay_deferred_grow()
        for round_no in range(2, 2 + patience):
            rt._round = round_no
            _ballot(relief, W4B_BACKED)
            rt._pay_deferred_grow()
            if rt._deferred_grow_rows == 0:
                break
        self.assertEqual(rt._deferred_grow_rows, 0)
        self.assertLessEqual(rt._round, 1 + patience)

    def test_a_settled_debt_stops_shouting(self):
        """A latched alarm for a condition that has gone is the indicator
        failure this queue has recorded eleven times in one day."""
        relief = _relief()
        rt = _runtime(relief)
        rt.book_deferred_grow(TP_TO_PP, W4B_AGREED)
        rt._round = 1
        rt._pay_deferred_grow()
        _ballot(relief, W4B_BACKED)
        rt._round = 1 + max(1, _rt.seam_shrink_grow_debt_rounds())
        with self.assertNoLogs("sglang.srt.managers.phase_flip_runtime", level="ERROR"):
            rt._deferred_grow_debt_check()
        self.assertEqual(rt._deferred_grow_rows, 0)

    def test_payment_never_takes_itself_back(self):
        """The booked level is a FLOOR under the clamp, never a ceiling.

        ``_pay_deferred_grow`` clamps to the level booked when the deferral was
        made. If the group has since agreed a HIGHER one, clamping to the
        booking undoes the payment and re-books the same debt -- the ratchet
        with an extra step.
        """
        relief = _relief()
        rt = _runtime(relief)
        rt.book_deferred_grow(TP_TO_PP, W4B_AGREED)
        _ballot(relief, W4B_BACKED)
        self.assertEqual(relief.exposed_rows(), W4B_BACKED)
        rt._round = 1
        rt._pay_deferred_grow()
        self.assertEqual(
            relief.exposed_rows(),
            W4B_BACKED,
            "the payment clamped back down to the stale booked level",
        )


class TheDebtStillRefusesRatherThanGuesses(CustomTestCase):
    """Paid OR named. Never silently accumulated, never silently exposed."""

    def test_no_agreed_level_exposes_nothing_and_says_so(self):
        """The #792 decline arriving at the payment. Rows stay invisible."""
        relief = _relief()
        rt = _runtime(relief)
        rt.book_deferred_grow(TP_TO_PP, None)
        rt._round = 1
        rt._pay_deferred_grow()
        self.assertEqual(
            relief.exposed_rows(),
            W4B_RESERVATION,
            "no cap was ever engaged, so the id space is the reservation -- "
            "this pins that the payment did not invent a level of its own",
        )
        self.assertIsNone(rt._deferred_grow_level)

    def test_a_ballot_pays_a_debt_that_had_no_agreed_level(self):
        """The decline is not permanent: the first honest verdict settles it."""
        relief = _relief()
        rt = _runtime(relief)
        rt.book_deferred_grow(TP_TO_PP, None)
        rt._round = 1
        rt._pay_deferred_grow()
        _ballot(relief, W4B_BACKED)
        rt._round = 2
        rt._deferred_grow_debt_check()
        self.assertEqual(rt._deferred_grow_rows, 0)

    def test_the_debt_reading_comes_from_the_allocator(self):
        """READ FROM THE POOL, NOT FROM THE BOOKING.

        The booking says what this runtime intended; the allocator says what
        admission is really priced against. #834's own docstring says to re-read
        and then read the booking anyway, which is how a settled debt kept
        shouting.
        """
        relief = _relief()
        rt = _runtime(relief)
        rt.book_deferred_grow(TP_TO_PP, W4B_AGREED)
        rt._round = 1
        rt._pay_deferred_grow()
        _ballot(relief, W4B_BACKED)
        # The BOOKING is untouched and still names the old level.
        self.assertEqual(rt._deferred_grow_level, W4B_AGREED)
        # The debt is nevertheless zero, because the pool says so.
        self.assertEqual(
            rt._unlevelled_rows(relief, rt._deferred_grow_level),
            0,
        )


class ThePaymentPathIsWired(CustomTestCase):
    """Wired, not merely written -- and reachable off the cutover.

    #834 shipped a debt with a deadline and no creditor. The structural claim
    that closes it is that the payment is reachable from the rung's per-round
    reduction, which every rank enters unconditionally, and NOT only from the
    seam's post-cutover hook.
    """

    def test_the_rung_reduction_publishes_the_group_level(self):
        import inspect

        from sglang.srt.managers import phase_flip_spill

        src = inspect.getsource(phase_flip_spill.collective_kv_backing_relief)
        self.assertIn(
            "publish_group_exposure",
            src,
            "the only actuator that raises the agreed level is still the "
            "seam's post-cutover hook, so the debt has no creditor",
        )

    def test_the_publication_enters_no_new_collective(self):
        """The constraint that ruled out moving the levelling itself.

        ``publish_group_exposure`` must act on a value the caller already has;
        the moment it reaches for a channel of its own it becomes a blocking
        reduction at a rank-local cadence, which is the 2026-08-08 wedge.
        """
        import inspect

        from sglang.srt.managers.kv_backing_relief import KvBackingRelief

        src = inspect.getsource(KvBackingRelief.publish_group_exposure)
        for forbidden in ("reduce_fn", "all_reduce", "broadcast", "barrier"):
            self.assertNotIn(
                forbidden,
                src.split('"""')[-1],
                f"the publication reaches for {forbidden}, which makes it a "
                "collective at a rank-local cadence",
            )


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
