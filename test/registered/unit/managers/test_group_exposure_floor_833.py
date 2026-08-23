"""#833 -- the exposed id space must be IDENTICAL across the group.

THE REGRESSION THIS CLOSES, measured on boot_window3_0823_1733.

That boot completed seven cutovers in its first two minutes and then never
flipped again for the remaining 22 of its 32 minutes. Every ``tp_to_pp``
attempt from 17:43:12 onward abandoned on the frame ballot, and all twelve
abandon lines (4 attempts x 3 ranks) named the same number::

    12 x "the poorest rank has only 120832 rows BACKED"

The window report framed this as a backing-dial LATCH on PP1. It is not one.
PP1's zero-byte release is already detected and already explained by the tree,
in the very next log line::

    KV-BACKING shrink to 114287 rows released NOTHING and the pool agrees:
    ... asked 6545 rows against a release granularity of 8192 rows
    (commit chunk 8 MiB across 32 buffers)

That is granularity, correctly reported. PP1 then took no second grow call for
an equally correct reason -- it had released nothing, so recovery had nothing
to restore, while PP0 and PP2 were ratcheted to their reservations by the #684
clamp. And 120832 is not a stuck value at all: it is PP1's CEILING. The three
ranks' reservations that boot were::

    PP0 reserved_backing_rows=204334
    PP1 reserved_backing_rows=119782      <- the narrowest, by design
    PP2 reserved_backing_rows=126828

Unequal BY DESIGN: uneven token vector 29,19,16 and TP vector 32,16,16 were
both on, as the standing rule requires them to be on every boot.

THE ACTUAL DEFECT IS ONE LEVEL UP, and #816's own acceptance evidence records
it without naming it. All three ranks enter the exposure clamp at ONE id space
and leave at THREE::

    boot 0516  PP2 exposed 449306  committed 126976  withdrew 322330
               PP1 exposed 449306  committed 120832  withdrew 328474
               PP0 exposed 449306  committed 204800  withdrew 244506

``exposed_rows`` states the contract that breaks here, in its own docstring:
the exposed id space "has to be identical across the group ... it decides
which ids the flip's live-slot enumeration can encounter". #816 closed a real
crash -- four boots died on the device-side assert -- by capping each rank at
its OWN committed backing. Under unequal pools that makes the exposures
differ, so the widest rank hands out ids the narrowest has no page for, and
``_agree_live_slots`` must refuse the union the moment the live id space grows
past the narrowest backing. Sustained load re-issues high ids faster than they
drain, so the refusal never lifts.

WHY WINDOW 2 PASSED THIS AND WINDOW 3 FAILED IT. Window 2 died after 75 s, at
0 of 9 latched dial calls, before its id space could reach the floor. The
regression did not appear between the two trees; the second boot merely lived
long enough to reach it. A criterion that can only fail on a long boot is not
evidence of health on a short one.

THE LAW THIS ANSWERS TO. Per the standing rule, a finding of the form "rank A
has surplus but it is unreachable because rank B binds" is ALWAYS a defect
report and NEVER a capacity verdict. PP0's 204800 rows are real. Reaching them
is the standing #795 federation debt. What is not acceptable in the meantime
is issuing ids no consumer can honour and then discovering it at the seam,
once per flip, forever.
"""

import unittest

import torch

from sglang.srt.managers.kv_backing_relief import (
    GROUP_FLOOR_UNKNOWN,
    collective_slot_ballot,
    group_exposure_ceiling,
    slot_proposal,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8)

BYTES_PER_ROW = 4096
LAW_FLOOR = 1024 * 1024 * 1024


class _FakeAlloc:
    """Just enough allocator for ``KvRowCap`` to hold ids back."""

    def __init__(self, size: int):
        self.size = int(size)
        self.free_pages = torch.arange(1, int(size) + 1, dtype=torch.int64)
        self.residency_withheld_slots = 0


class _FakeVmmPool:
    """A VMM-backed pool whose committed rows and id space can diverge."""

    def __init__(self, backed_rows: int, reserved_rows: int, page_size: int = 1):
        self.full_pool_backed_rows = int(backed_rows)
        self.reserved = int(reserved_rows)
        self.reserved_backing_rows = int(reserved_rows)
        self.size = int(reserved_rows)
        self.page_size = int(page_size)
        self.attempts = []

    def runtime_set_backing_rows(self, rows: int) -> None:
        self.attempts.append(int(rows))
        self.full_pool_backed_rows = int(rows)


def _relief(pool, *, alloc=None, free_mib: int = 8192, live_rows=()):
    from sglang.srt.managers.kv_backing_relief import KvBackingRelief

    return KvBackingRelief(
        pool,
        allocator=alloc,
        live_slots_fn=lambda: (
            torch.tensor(list(live_rows), dtype=torch.int64)
            if live_rows
            else torch.empty((0,), dtype=torch.int64)
        ),
        bytes_per_row=BYTES_PER_ROW,
        probe=lambda: free_mib * (1 << 20),
        device_index=0,
        buffers=1,
        law_floor_bytes=LAW_FLOOR,
        pool_fn=lambda: None,
    )


#: The three ranks of boot_window3_0823_1733, at 17:41:40, verbatim.
W3_BACKED = {"PP0": 204800, "PP1": 120832, "PP2": 126976}
#: The id space all three ranks exposed BEFORE any clamp, from #816's own
#: acceptance evidence for the same three-rank shape.
W3_EXPOSED_BEFORE_CLAMP = 449306
#: The highest row the group's union reached on the first refused attempt.
W3_UNION_HIGH = 185290


def _min_reduce(proposals):
    """The MIN all-reduce the rung runs, as a pure list operation."""
    return [min(fields[i] for fields in proposals) for i in range(len(proposals[0]))]


def _union_refused(union_high, min_backed):
    """The frame ballot's bound, phase_flip_runtime.py `_agree_live_slots`."""
    return int(union_high) >= int(min_backed)


class TestGroupExposureCeiling(unittest.TestCase):
    """The pure decision, with no pool, no arena and no boot."""

    def test_unknown_floor_is_816_behaviour_unchanged(self):
        """No group verdict seen -> the local backing, exactly as before.

        This is the clause that keeps every rank-local path (a recovery with
        no collective, a stub runtime, a hermetic test) behaving as it did.
        """
        for backed in (0, 1, 120832, 204800):
            self.assertEqual(
                group_exposure_ceiling(backed, GROUP_FLOOR_UNKNOWN), backed
            )

    def test_zero_floor_is_a_real_floor_not_unknown(self):
        """A floor of 0 must not collapse into 'unknown'.

        That sentinel collision is what #714 paid for, where "none" and
        "unknown" shared -1 and eviction pricing silently skipped.
        """
        self.assertEqual(group_exposure_ceiling(204800, 0), 0)
        self.assertNotEqual(
            group_exposure_ceiling(204800, 0),
            group_exposure_ceiling(204800, GROUP_FLOOR_UNKNOWN),
        )

    def test_ceiling_never_raises_local_exposure(self):
        """A floor ABOVE this rank's backing may not lift it.

        Exposing above the committed backing is the #816 crash; the group
        floor is allowed to lower this rank's ceiling and never to raise it.
        """
        self.assertEqual(group_exposure_ceiling(120832, 204800), 120832)

    def test_the_window3_group_levels_to_one_id_space(self):
        """The measured three ranks reach ONE exposure, which is the contract."""
        floor = min(W3_BACKED.values())
        ceilings = {
            rank: group_exposure_ceiling(backed, floor)
            for rank, backed in W3_BACKED.items()
        }
        self.assertEqual(set(ceilings.values()), {120832})


class TestUnequalPoolsWedgeTheFlip(unittest.TestCase):
    """The causal chain, end to end, on the measured numbers.

    RED-FIRST: every assertion here fails on the unfixed behaviour, and the
    unfixed behaviour is expressed as `group_exposure_ceiling(backed,
    GROUP_FLOOR_UNKNOWN)` -- i.e. clamping to the local backing, which is
    literally what #816 does. So the danger direction is not a hypothetical:
    putting the local-only clamp back turns these red again.
    """

    def _group_floor(self):
        """What the rung's own reduction yields for the measured group."""
        reduced = _min_reduce(
            [
                slot_proposal(digest=7, max_live_row=W3_UNION_HIGH, backed_rows=b)
                for b in W3_BACKED.values()
            ]
        )
        ballot = collective_slot_ballot(reduced)
        self.assertIsNotNone(ballot)
        return int(ballot["min_backed_rows"])

    def test_the_reduction_reports_the_narrowest_rank(self):
        self.assertEqual(self._group_floor(), 120832)

    def test_local_clamp_diverges_the_id_space(self):
        """THE DEFECT. Local-only clamping gives three different exposures."""
        exposures = {
            rank: group_exposure_ceiling(
                min(W3_EXPOSED_BEFORE_CLAMP, backed), GROUP_FLOOR_UNKNOWN
            )
            for rank, backed in W3_BACKED.items()
        }
        # Three ranks entered at ONE exposure (449306) and left at THREE.
        self.assertEqual(len(set(exposures.values())), 3)
        self.assertEqual(exposures, {"PP0": 204800, "PP1": 120832, "PP2": 126976})

    def test_divergent_exposure_lets_a_peer_issue_an_unmappable_id(self):
        """The widest rank can hand out an id the narrowest cannot map."""
        floor = self._group_floor()
        widest = group_exposure_ceiling(W3_BACKED["PP0"], GROUP_FLOOR_UNKNOWN)
        self.assertGreater(widest, floor)
        # An id in (floor, widest] exists on PP0 and is unmapped on PP1.
        self.assertTrue(_union_refused(union_high=widest - 1, min_backed=floor))

    def test_levelled_exposure_cannot_build_a_refusable_union(self):
        """THE FIX. With one id space, no issuable id can breach the floor.

        This is the load-bearing claim: it is not that the union happens to
        fit this round, it is that a set which the ballot must refuse can no
        longer be CONSTRUCTED, because no rank can issue an id above the
        floor in the first place.
        """
        floor = self._group_floor()
        ceilings = [
            group_exposure_ceiling(backed, floor) for backed in W3_BACKED.values()
        ]
        self.assertEqual(len(set(ceilings)), 1)
        highest_issuable_id = max(ceilings) - 1
        self.assertFalse(
            _union_refused(union_high=highest_issuable_id, min_backed=floor)
        )

    def test_the_measured_refusal_reproduces_and_the_fix_removes_it(self):
        """The exact abandon line of 17:43:12, both directions."""
        floor = self._group_floor()
        # Unfixed: PP0 could expose 204800, so a union reaching 185290 exists.
        self.assertTrue(_union_refused(W3_UNION_HIGH, floor))
        self.assertLessEqual(
            W3_UNION_HIGH, group_exposure_ceiling(W3_BACKED["PP0"], GROUP_FLOOR_UNKNOWN)
        )
        # Fixed: 185290 is above every rank's ceiling, so no rank issues it.
        levelled = group_exposure_ceiling(W3_BACKED["PP0"], floor)
        self.assertGreater(W3_UNION_HIGH, levelled)


class TestStrandedSurplusIsNamedNotHidden(unittest.TestCase):
    """The cost is real and must be countable, per the standing law."""

    def test_the_surplus_is_reported_as_a_quantity(self):
        floor = min(W3_BACKED.values())
        stranded = {
            rank: backed - group_exposure_ceiling(backed, floor)
            for rank, backed in W3_BACKED.items()
        }
        self.assertEqual(stranded, {"PP0": 83968, "PP1": 0, "PP2": 6144})
        # The narrowest rank strands nothing -- it IS the floor.
        self.assertEqual(stranded["PP1"], 0)
        # And the surplus is not a rounding error; it is 41% of PP0's pool.
        self.assertGreater(stranded["PP0"], W3_BACKED["PP0"] // 3)


class TheClampHonoursTheGroupFloor(unittest.TestCase):
    """The actuator, not just the arithmetic: does the clamp USE the floor.

    A pure function nobody calls is the "wired and inert" failure this queue
    has recorded repeatedly, so each direction is exercised on a real
    ``KvBackingRelief``.
    """

    def _pp0(self, live_rows=()):
        """The widest rank of boot_window3: 204800 backed, 449306 exposed."""
        pool = _FakeVmmPool(
            backed_rows=W3_BACKED["PP0"], reserved_rows=W3_EXPOSED_BEFORE_CLAMP
        )
        return _relief(
            pool,
            alloc=_FakeAlloc(W3_EXPOSED_BEFORE_CLAMP),
            live_rows=live_rows,
        )

    def test_without_a_floor_it_caps_at_the_local_backing(self):
        """#816's behaviour, unchanged where no group verdict exists."""
        relief = self._pp0()
        withdrew = relief.clamp_exposure_to_backing("hermetic")
        self.assertEqual(withdrew, W3_EXPOSED_BEFORE_CLAMP - W3_BACKED["PP0"])
        self.assertEqual(relief.exposed_rows(), W3_BACKED["PP0"])

    def test_with_a_floor_it_caps_at_the_group_level(self):
        """THE FIX, at the actuator: PP0 stops exposing what PP1 cannot map."""
        relief = self._pp0()
        relief.note_group_backing_floor(W3_BACKED["PP1"])
        withdrew = relief.clamp_exposure_to_backing("hermetic")
        self.assertEqual(relief.exposed_rows(), W3_BACKED["PP1"])
        self.assertEqual(withdrew, W3_EXPOSED_BEFORE_CLAMP - W3_BACKED["PP1"])

    def test_the_clamp_still_never_lowers_the_backing(self):
        """The group floor bounds the ID SPACE and never the backing.

        Lowering the backing here would be #722 -- a cap under live rows --
        which is the crash #816 mirrors. The pool must see no dial call.
        """
        relief = self._pp0()
        relief.note_group_backing_floor(W3_BACKED["PP1"])
        relief.clamp_exposure_to_backing("hermetic")
        self.assertEqual(relief._pool.attempts, [])
        self.assertEqual(relief._pool.full_pool_backed_rows, W3_BACKED["PP0"])
        # And the rank still REPORTS its true backing to the group, so the
        # next reduction is not fed a value this clamp invented.
        self.assertEqual(relief.backed_rows(), W3_BACKED["PP0"])

    def test_an_abstain_sentinel_does_not_withdraw_the_id_space(self):
        """A peer that could not report must not zero this rank's exposure.

        The abstain and truncated-payload cases both arrive negative; reading
        either as a floor of 0 would strand the entire pool on a rank whose
        peer merely failed to answer.
        """
        relief = self._pp0()
        relief.note_group_backing_floor(-1)
        relief.clamp_exposure_to_backing("hermetic")
        self.assertEqual(relief.exposed_rows(), W3_BACKED["PP0"])

    def test_live_rows_between_floor_and_backing_are_not_a_722_alarm(self):
        """A live row above the group floor is MAPPED here -- not #722.

        Testing the #722 condition against the group ceiling instead of the
        committed backing would raise a false unmapped-live-rows error on
        every rank wider than the group's poorest, i.e. on almost every rank.
        """
        live = W3_BACKED["PP1"] + 1000
        relief = self._pp0(live_rows=(live,))
        relief.note_group_backing_floor(W3_BACKED["PP1"])
        with self.assertLogs(
            "sglang.srt.managers.kv_backing_relief", level="WARNING"
        ) as caught:
            relief.clamp_exposure_to_backing("hermetic")
        self.assertFalse(
            [line for line in caught.output if "#722 state underneath" in line],
            "a live row inside this rank's own backing is not the #722 crash",
        )


class TheBallotFeedsTheClamp(unittest.TestCase):
    """The wire itself: the rung's reduction must reach the exposure clamp.

    #833's whole content is that these two already-existing quantities were
    never connected. A test on the pure function alone would pass on the
    unfixed tree.
    """

    def test_the_seam_funding_verdict_notes_the_floor(self):
        """The rung names the entry point, in whichever form it calls it.

        The call is guarded (``getattr``) so a stub rung or a peer on an older
        tree cannot take a cutover down with an AttributeError, so this looks
        for the NAME rather than for one syntactic shape -- an assertion keyed
        to `ast.Attribute` alone would have gone green on the unfixed tree the
        moment the call was made defensive.
        """
        import ast
        import inspect
        import textwrap

        from sglang.srt.managers import phase_flip_spill

        tree = ast.parse(textwrap.dedent(inspect.getsource(phase_flip_spill)))
        named = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn(
            "note_group_backing_floor",
            named,
            "the group floor is reduced here and must be handed to the clamp; "
            "without this call the pure function is wired-and-inert",
        )

    def test_a_rung_without_the_entry_point_is_announced_not_swallowed(self):
        """An inert path must say so -- the W9 '105 silent fallbacks' lesson."""
        import inspect

        from sglang.srt.managers import phase_flip_spill

        src = inspect.getsource(phase_flip_spill.collective_kv_backing_relief)
        self.assertIn("cannot take the group backing floor", src)

    def test_the_relief_exposes_the_entry_point(self):
        self.assertTrue(
            hasattr(_relief(_FakeVmmPool(10, 20)), "note_group_backing_floor")
        )


if __name__ == "__main__":
    unittest.main()
