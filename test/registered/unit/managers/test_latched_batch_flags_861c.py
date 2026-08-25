# SPDX-License-Identifier: Apache-2.0
"""#861c: latched batch flags across the #856 seam — the CLASS, not the third instance.

CLASS: **a persistent flag on a ScheduleBatch whose only clear sites are FINISH
paths.** The #856 seam RETRACTS residents instead of finishing them, so nothing
on those paths runs and the flag survives into the next phase on a batch that
now holds zero requests.

THREE MEMBERS, and the third is the point:

    spec_algorithm  -> retune_carried_batches_for_phase   (#631, fixed)
    spec_info       -> clear_spec_info_for_unspeculated_phase (#631, fixed)
    batch_is_full   -> reset_stale_batch_flags            (W37-C, fixed here)

The first two each got a bespoke handler and the CLASS was never named, which
is exactly why the third went unnoticed for a whole window. W37-C measured it:
``batch_is_full=1`` with ``running=0``, ``avail=468981``, ``evictable=0`` --
admission declining for ever, because a batch that is full with zero running
requests can never drain.

SIBLINGS SWEPT: `grep -c batch_is_full phase_flip_runtime.py` was **0**; the
clear sites are scheduler.py:6488/6507/6562, all finish paths.

FUTURE-CHECK: `STALE_BATCH_FLAGS` is a declared table, `reset_stale_batch_flags`
walks the SAME `_reachable_batches` its two siblings use, and
`test_a_fourth_member_would_be_handled_automatically` pins that adding a row to
the table is all a fourth member needs.
"""

import types


from sglang.srt.managers.phase_flip_draft_bootstrap import (
    STALE_BATCH_FLAGS,
    reset_stale_batch_flags,
)


class FakeBatch:
    def __init__(self, batch_is_full=False, reqs=None):
        self.batch_is_full = batch_is_full
        self.reqs = reqs if reqs is not None else []

    def is_empty(self):
        return not self.reqs


def sched_with(**batches):
    base = dict(
        running_batch=None, last_batch=None, cur_batch=None, running_mbs=None
    )
    base.update(batches)
    return types.SimpleNamespace(**base)


def test_the_specimen_a_full_batch_with_zero_running_requests():
    """W37-C's exact state: latched True, no reqs. Nothing will ever free a
    slot, so admission declines for ever unless the seam clears it."""
    rb = FakeBatch(batch_is_full=True, reqs=[])
    cleared = reset_stale_batch_flags(sched_with(running_batch=rb))
    assert cleared["batch_is_full"] == 1
    assert rb.batch_is_full is False


def test_it_reaches_last_batch_too():
    """`last_batch` is a live merge target the next round can still reach --
    the handle that killed PP0 at 2026-08-09 20:31:48 (corpse I). The reach is
    deliberately wider than the resident harvest."""
    lb = FakeBatch(batch_is_full=True)
    cleared = reset_stale_batch_flags(sched_with(last_batch=lb))
    assert cleared["batch_is_full"] == 1
    assert lb.batch_is_full is False


def test_it_reaches_running_mbs_slots():
    mb = FakeBatch(batch_is_full=True)
    cleared = reset_stale_batch_flags(sched_with(running_mbs=[mb, None]))
    assert cleared["batch_is_full"] == 1


def test_an_alias_is_counted_once():
    """`running_batch` is routinely an ALIAS of a `running_mbs` slot; counting
    the same object twice would misreport the number the seam logs."""
    b = FakeBatch(batch_is_full=True)
    cleared = reset_stale_batch_flags(sched_with(running_batch=b, running_mbs=[b]))
    assert cleared["batch_is_full"] == 1


def test_a_healthy_batch_is_not_touched_and_reports_zero():
    """The seam must log a NUMBER, not an intention: 0 means nothing was stale,
    and that has to be distinguishable from 'the reach missed them'."""
    b = FakeBatch(batch_is_full=False)
    cleared = reset_stale_batch_flags(sched_with(running_batch=b))
    assert cleared["batch_is_full"] == 0
    assert b.batch_is_full is False


def test_an_empty_scheduler_is_a_free_no_op():
    assert reset_stale_batch_flags(sched_with()) == {"batch_is_full": 0}


def test_a_batch_missing_the_attribute_entirely_is_safe():
    """Stand-ins in this tree carry only the fields the code under test reads."""
    b = types.SimpleNamespace(reqs=[])
    reset_stale_batch_flags(sched_with(running_batch=b))
    assert not hasattr(b, "batch_is_full"), (
        "the reset must not INVENT a flag on a batch that never had one -- "
        "that would create a fourth, silent binding (the `_stamp` rule)"
    )


def test_a_fourth_member_would_be_handled_automatically(monkeypatch):
    """FUTURE-CHECK: adding a row to the declared table is all it takes.

    This is what makes it a class fix rather than a third instance fix.
    """
    import sglang.srt.managers.phase_flip_draft_bootstrap as mod

    monkeypatch.setitem(mod.STALE_BATCH_FLAGS, "some_future_latch", False)
    b = FakeBatch(batch_is_full=True)
    b.some_future_latch = True
    cleared = reset_stale_batch_flags(sched_with(running_batch=b))
    assert cleared["some_future_latch"] == 1
    assert b.some_future_latch is False


def test_the_table_declares_the_default_not_a_computed_value():
    """The seam's job is to remove a STALE claim; the next ordinary round
    recomputes the honest one from the live pool. Guessing a value here would
    be a second opinion about admission -- the shape that produced the defect."""
    assert STALE_BATCH_FLAGS["batch_is_full"] is False


def test_can_fail_the_reset_is_not_a_tautology():
    """A reset that never writes would pass every 'is False' assertion above on
    a batch that started False. Pin it against one that starts True."""
    b = FakeBatch(batch_is_full=True)
    assert b.batch_is_full is True
    reset_stale_batch_flags(sched_with(running_batch=b))
    assert b.batch_is_full is False
