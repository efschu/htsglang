# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1204: a rank-local exit inside the seam's no-return region.

TWO SITES, ONE CLASS, AND THE CLASS IS "ONE RANK LEFT, THE OTHERS ARE STILL
WAITING". Both sit between the seam's no-return point and the group step that
follows it, and both were written as if leaving early were merely a degraded
outcome for the rank that left.

  D1  ``level_kv_backing_to_group`` reads ``backed_rows()`` and
      ``live_floor_rows()`` BEFORE it enters ``reduce_fn``, and it returns
      early when this rank has no relief object at all. Either way the rank
      walks on while its peers block for ever inside a blocking all-reduce.
      THE FAILURE PRODUCES NO TRACEBACK ON ANY RANK -- the leaver did not
      raise, and the ranks that hang are inside a collective, not inside
      Python. It is the one shape in this region that cannot be read out of a
      boot log at all.

      That this is reachable is not hypothetical. The relief is installed
      rank-locally (``phase_flip_spill.py``, ``kv_backing_provider`` under a
      ``try``/``except`` that sets ``relief = None`` on ANY per-device
      failure), and this rig is heterogeneous -- 31800 / 18800 / 19800 MiB per
      rank -- so a VMM failure on exactly one card is the ordinary shape of
      that ``except``, not an exotic one.

      The house rule is already written one screen down in the same file. The
      sibling ``collective_kv_backing_relief`` handles the identical
      abstention by setting ``relief = None`` and STILL REDUCING:

          # No floor to propose against. Abstain -- but still reduce, because a
          # rank that returns early here hangs the ones that did not.

  D2  the pre-cutover mover loop (``for fn in self._pre_cutover_fns``) runs the
      weights-arena refill and the GDN state leg unwrapped, inside the
      no-return region. It does not swallow -- so this one is loud rather than
      silent -- but what it raises is whatever the mover raised, with nothing
      saying WHICH mover, and nothing saying that it happened after the arena
      was already being rewritten. Serving from a half-refilled arena is the
      wrong-answer failure ``weights_arena.verify_boot_anchor`` exists to
      refuse; a mover failure here has to be as legible as that refusal is.

WHY THE CHANNEL IN THIS FILE COUNTS ARRIVALS. The stub channel the #792 fleet
ships computes ``min()`` and nothing else, which cannot express the property
under test: the production channel is a BLOCKING all-reduce, and the defect is
about WHO ENTERED IT, not about the value that comes back. So the channel here
records every arrival, and the assertions are about that list.
"""

import unittest

from sglang.srt.managers import phase_flip_runtime as pfr
from sglang.srt.managers import phase_flip_spill as pfs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_residency_cap_flip_levelling_792 import (  # noqa: E402 (sibling harness)
    POOL,
    _rank,
    _scheduler,
)

register_cpu_ci(est_time=5)

#: A live set far below the group minimum, so the levelling actually applies
#: rather than declining -- the same distinction the #834 split file draws.
LIVE_TOP_SMALL = 256
GROUP_MIN_LEVELLED = 2048


def _healthy(backed):
    return _rank(backed=backed, live_top=LIVE_TOP_SMALL, capped_at=None)


class _BlindRelief:
    """A relief whose per-device reads fail: the one-card VMM failure shape.

    It is a relief object, so the ``relief is None`` arm does not catch it --
    this is the OTHER pre-reduce exit, the one that lands in the ``except``
    whose message claims "a lost flip, never a half-flipped group".
    """

    def backed_rows(self):
        raise RuntimeError("#1204 specimen: cuMemGetAllocationGranularity failed")

    def live_floor_rows(self):  # pragma: no cover - never reached
        raise AssertionError("backed_rows() raises first")

    def level_recovery_to(self, level):  # pragma: no cover - never reached
        raise AssertionError("an abstaining rank must never apply a level")


class _CountingChannel:
    """The seam's element-wise MIN channel, plus the arrival ledger.

    ``group`` is precomputed from the ranks that CAN read their own numbers.
    That is exactly what a real MIN all-reduce returns once an abstaining rank
    contributes neutral sentinels, so the fixture does not assume the fix -- it
    states the group's answer independently and lets the assertions compare.
    """

    def __init__(self, healthy_fleet):
        gathered = [
            [r.backed_rows(), -r.backed_rows(), -r.live_floor_rows()]
            for r in healthy_fleet
        ]
        self.group = [min(vals) for vals in zip(*gathered)]
        self.arrivals = []

    def __call__(self, vals):
        vals = list(vals)
        self.arrivals.append(list(vals))
        return [min(mine, theirs) for mine, theirs in zip(vals, self.group)]


class TestD1AbstainIntoTheReduction(CustomTestCase):
    """Every rank that reaches the levelling must reach its reduction."""

    def _fleet_with(self, odd_scheduler_relief):
        """Two ranks that can read themselves, one that cannot, in rank order."""
        healthy = [_healthy(POOL), _healthy(GROUP_MIN_LEVELLED)]
        channel = _CountingChannel(healthy)
        schedulers = [_scheduler(healthy[0]), _scheduler(healthy[1])]
        schedulers.insert(1, odd_scheduler_relief)
        return channel, schedulers

    def test_red_a_a_rank_with_no_relief_still_enters_the_reduction(self):
        """THE SILENT WEDGE. No relief object, so nothing to level -- but the
        peers are already inside a blocking collective and the only thing that
        releases them is this rank arriving too."""
        no_relief = type("S", (), {})()
        channel, schedulers = self._fleet_with(no_relief)
        for sched in schedulers:
            pfs.level_kv_backing_to_group(sched, channel)
        self.assertEqual(
            3,
            len(channel.arrivals),
            "a rank that returns before the reduction hangs the ones that did "
            "not, and it raises nothing while doing so",
        )

    def test_red_b_a_rank_that_cannot_read_its_backing_still_enters(self):
        """The SECOND pre-reduce exit: ``backed_rows()`` raises, the ``except``
        arm catches it and returns. Same wedge, and this one leaves a log line
        that says the opposite of what happened."""
        channel, schedulers = self._fleet_with(_scheduler(_BlindRelief()))
        for sched in schedulers:
            pfs.level_kv_backing_to_group(sched, channel)
        self.assertEqual(3, len(channel.arrivals), "the except arm returns early")

    def test_red_c_the_abstention_is_neutral_to_the_group_decision(self):
        """THE DANGEROUS DIRECTION, and the reason a sentinel is not free.

        Arriving is only half of it: a rank that arrives carrying 0 (or -1, or
        its uninitialised backing) drags the group MIN down and the whole fleet
        agrees a level it must then cap itself to. That corrupts where the hang
        merely stops. So the payload an abstaining rank contributes must leave
        the element-wise MIN of the whole group identical to the MIN over the
        ranks that could actually read themselves."""
        channel, schedulers = self._fleet_with(_scheduler(_BlindRelief()))
        for sched in schedulers:
            pfs.level_kv_backing_to_group(sched, channel)
        self.assertEqual(3, len(channel.arrivals))
        everyone = [min(v) for v in zip(*channel.arrivals)]
        self.assertEqual(
            channel.group,
            everyone,
            "the abstaining rank's payload must be neutral in EVERY slot of "
            "the MIN channel, floor included",
        )

    def test_green_the_abstaining_rank_reports_no_level(self):
        """It agreed to nothing it can apply, so it must not report a level a
        deferred grow would then clamp its exposure to."""
        channel, schedulers = self._fleet_with(_scheduler(_BlindRelief()))
        levels = [pfs.level_kv_backing_to_group(s, channel) for s in schedulers]
        self.assertIsNone(levels[1], "an abstainer has no level to hand out")
        self.assertEqual(
            [GROUP_MIN_LEVELLED, GROUP_MIN_LEVELLED],
            [levels[0], levels[2]],
            "the peers still agree the real group minimum",
        )

    def test_green_no_channel_at_all_returns_without_reducing(self):
        """THE ONE EARLY RETURN THAT STAYS. No ``reduce_fn`` means there is no
        collective for anyone to be waiting in -- single-rank shapes -- so
        returning is correct and must not become a call on None."""
        self.assertIsNone(
            pfs.level_kv_backing_to_group(_scheduler(_healthy(POOL)), None)
        )


class _HalfBlindRelief:
    """Backing readable, floor NOT: the two reads shared one ``try``.

    The relief object is installed and this rank really did measure its own
    backing. Only the SECOND read fails -- and the two reads sat in one
    ``try``, so the ``except`` threw the measured backing away and the rank
    abstained in every slot.
    """

    def __init__(self, backed):
        self._backed = int(backed)

    def backed_rows(self):
        return self._backed

    def live_floor_rows(self):
        raise RuntimeError("#1204 repair specimen: the FLOOR read failed")

    def level_recovery_to(self, level):  # pragma: no cover - abstainer
        raise AssertionError("a rank that could not read its floor levels nothing")


class TestD1AMeasuredTermIsNeverDiscarded(CustomTestCase):
    """REPAIR of #1204: abstain PER SLOT, not per rank.

    THE SENTINEL IS CORRECT FOR WHAT IT WAS WRITTEN FOR and wrong for what it
    was applied to. ``_UNBOUNDED_ROWS`` in slot 0 of a MIN says "I am not the
    poorest" -- true of a rank with no cap machinery at all, and a CLAIM a
    rank that just measured a poor backing has no right to make.

    Slot 0's MIN is the group's poorest backing, and it becomes both the level
    the peers cap to and the ceiling ``book_deferred_grow`` clamps a later grow
    up to. Erasing a poor rank from that MIN makes the group agree an id level
    that rank cannot map -- which ``backed_rows``' own docstring calls "a
    ``cudaErrorIllegalAddress`` that kills every rank rather than raising".
    The hang the commit removed was traded for a wrong group answer, which is
    the corrupting direction, not the refusing one.
    """

    PEERS = (100, 120)
    POOR = 30

    def _run(self, poor_relief):
        healthy = [_healthy(b) for b in self.PEERS]
        channel = _CountingChannel(healthy)
        schedulers = [
            _scheduler(healthy[0]),
            _scheduler(poor_relief),
            _scheduler(healthy[1]),
        ]
        levels = [pfs.level_kv_backing_to_group(s, channel) for s in schedulers]
        return channel, levels

    def test_a_readable_backing_reaches_the_group_min(self):
        """THE DEFECT, one assertion. The floor read failed; the BACKING did
        not, and the group's poorest must still be this rank."""
        channel, _ = self._run(_HalfBlindRelief(self.POOR))
        self.assertEqual(3, len(channel.arrivals), "every rank must still arrive")
        mine = channel.arrivals[1]
        self.assertEqual(
            self.POOR,
            mine[0],
            "this rank measured its backing and then contributed the MIN "
            "identity for it -- it told the group it is not the poorest",
        )
        group_min = min(a[0] for a in channel.arrivals)
        self.assertEqual(
            self.POOR,
            group_min,
            "the peers agreed a level above the rows this rank can map",
        )

    def test_an_unreadable_floor_still_declines_rather_than_agreeing(self):
        """An unknown floor is not a low one -- the function's own rule for a
        truncated channel, applied to a locally unreadable one."""
        _, levels = self._run(_HalfBlindRelief(self.POOR))
        self.assertIsNone(
            levels[1],
            "a rank that could not read its floor may not report a level a "
            "deferred grow would clamp its exposure to",
        )

    def test_a_wholly_unreadable_relief_is_still_neutral(self):
        """The other trigger is unchanged: nothing measured, nothing claimed
        beyond the neutral the channel already uses for it."""
        healthy = [_healthy(b) for b in self.PEERS]
        channel = _CountingChannel(healthy)
        pfs.level_kv_backing_to_group(_scheduler(_BlindRelief()), channel)
        self.assertEqual(1, len(channel.arrivals))
        self.assertEqual(
            [min(v) for v in zip(channel.arrivals[0], channel.group)],
            channel.group,
            "a rank that measured nothing must stay neutral in every slot",
        )


class TestD2PreCutoverMoversAreNamed(CustomTestCase):
    """A mover that fails inside the no-return region says so, by name."""

    @staticmethod
    def _mover(label, boom=None):
        def fn(direction):
            if boom is not None:
                raise boom

        fn.census_label = label
        return fn

    def test_green_every_mover_runs_and_is_marked_in_order(self):
        marks = []
        ran = []
        fns = (
            self._mover("weights_refill"),
            self._mover("gdn_state"),
        )
        pfr.run_pre_cutover_movers(
            tuple(self._wrap(fn, ran) for fn in fns), "tp_to_pp", marks.append
        )
        self.assertEqual(["weights_refill", "gdn_state"], marks)
        self.assertEqual(["tp_to_pp", "tp_to_pp"], ran)

    @staticmethod
    def _wrap(fn, ran):
        def wrapped(direction):
            ran.append(direction)
            return fn(direction)

        wrapped.census_label = fn.census_label
        return wrapped

    def test_red_a_failure_raises_a_named_error_that_names_the_mover(self):
        """NOT SWALLOWED, and not anonymous either. The raw error says nothing
        about which leg mutated what, and the arena refill and the GDN state
        leg have opposite consequences."""
        fns = (
            self._mover("weights_refill", boom=RuntimeError("cuMemMap failed")),
            self._mover("gdn_state"),
        )
        with self.assertRaises(pfr.SeamMoverError) as caught:
            pfr.run_pre_cutover_movers(fns, "pp_to_tp", lambda label: None)
        self.assertIn("weights_refill", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_red_b_a_failed_mover_is_not_marked_and_stops_the_loop(self):
        """The census must not record a leg that did not finish, and the legs
        behind a failed one must not run over a half-written arena."""
        marks = []
        ran = []
        fns = (
            self._mover("weights_refill", boom=RuntimeError("cuMemMap failed")),
            self._wrap(self._mover("gdn_state"), ran),
        )
        with self.assertRaises(pfr.SeamMoverError):
            pfr.run_pre_cutover_movers(fns, "pp_to_tp", marks.append)
        self.assertEqual([], marks)
        self.assertEqual([], ran)

    def test_red_c_the_failure_is_recorded_before_it_is_raised(self):
        """The rank dies on the raise, so whatever a post-mortem reads has to
        have been written BEFORE it -- the label, not just the traceback."""
        noted = []
        fns = (self._mover("gdn_state", boom=RuntimeError("state move failed")),)
        with self.assertRaises(pfr.SeamMoverError):
            pfr.run_pre_cutover_movers(
                fns,
                "tp_to_pp",
                lambda label: None,
                note_failure=lambda label, exc: noted.append((label, str(exc))),
            )
        self.assertEqual(1, len(noted))
        self.assertEqual("gdn_state", noted[0][0])


if __name__ == "__main__":
    unittest.main()
