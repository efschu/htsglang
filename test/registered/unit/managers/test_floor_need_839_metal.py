"""#839-METAL -- the seam ballot knows the live-set need and nothing consumes it.

THE SPECIMEN, window 5, integ/round5 @ 54e69ca2af, BOTH boots, identical to the
row (/spinning/evidence-665-f1/window5_radix_sanity_crash/)::

    live set needs 131073 (this rank backs 126976, its own floor is 131073)
    live set needs 131073 (this rank backs 133120, its own floor is 131073)
    live set needs 131073 (this rank backs 215040, its own floor is 131073)

    agreed level = 126976            GROW-DEBT-UNPAID standing (1368 / 1916)
    GROW DEFERRED 3 / GROW PAID 2    153 flip arms -> 0 tp_to_pp completions
    "[#839] exposure published (seam ballot)"  =  0 lines on both boots

WHAT THE ZERO DOES NOT MEAN. ``publish_group_exposure`` logs only ``if moved:``
and has FIVE exits, four of which return 0 in silence. A hermetic replay of the
numbers above through the REAL actuator shows the ballot running on every round
and every rank moving 0 -- so the metal zero is a path that EXECUTED on all 153
arms and declined, not a path that was never reached. The "publication needs a
flip, the flip needs publication" cycle is FALSIFIED: the publication rides
``collective_kv_backing_relief`` from ``_corridor_gate``
(phase_flip_runtime.py:8041), which ran at least 102 times on 5b -- it printed
its own refusal that many times.

THE ACTUAL ROOT, and it is an arithmetic property, not a wiring gap. The agreed
level is ``min(backed)`` over the group. Paying EVERY deferred grow on the two
ranks that carry debt (PP0 +88064, PP2 +6144) moves the level by EXACTLY ZERO,
because the floor is PP1 and PP1 carries no debt. The debt of a rank that is not
the floor is structurally unpayable by publication, so #834 crit 13's counter
cannot reach zero this way -- which is what 1368 standing UNPAID lines say.

The one number that would unblock the group is the poorest rank growing 4097
rows, and THAT NUMBER IS ALREADY ON THE WIRE. ``collective_slot_ballot``
(kv_backing_relief.py:537-543) decodes BOTH from the same reduced payload::

    "max_live_row":   -int(reduced[2])    the id space a union has to span
    "min_backed_rows": int(reduced[3])    the highest row EVERY rank has backed

and phase_flip_spill.py:2044 consumes ``min_backed_rows`` alone. The need is
computed, reduced, decoded and dropped.

THE DANGER DIRECTION, pinned below and not merely asserted in prose: growing a
rank COMMITS PAGES and must NEVER announce them. #839 A's rule is untouched by
this change -- exposure still rises only in ``publish_group_exposure``, only on
a group verdict, only in the arena that verdict was measured in. A grow makes
the NEXT ballot's floor higher; it never raises this rank's exposure itself.
``test_growing_the_floor_rank_never_raises_its_own_exposure`` is the guard, and
it is written to fail if a future edit takes the shortcut.
"""

import unittest

import torch

BYTES_PER_ROW = 32768
LAW_FLOOR = 1024 * 1024 * 1024

#: Window 5, from the abandon lines, to the row. PP1 is the floor and the
#: binding rank; PP0 and PP2 are the ranks carrying deferred-grow debt.
W5_BACKED = {"PP0": 215040, "PP1": 126976, "PP2": 133120}
#: "live set needs 131073", on every abandon line of both boots. This is the
#: SPAN -- the number of rows a union must cover.
W5_LIVE_NEED = 131073
#: What the BALLOT actually carries, and therefore what the callsite hands the
#: rung: the highest live ROW ID, one less than the span
#: (``span = int(ballot.get("max_live_row", -1)) + 1``,
#: phase_flip_runtime.py:8754).
#:
#: #839-METAL v2 CONTRACT CHANGE, stated rather than quietly absorbed: v1's
#: ``note_group_live_need`` treated its argument AS the span, which is the
#: off-by-one v2 fixes -- it asked for one row less than the flip needs. These
#: tests fed it the span directly and so did not see the defect. They now feed
#: the row id, like the real callsite does, and assert the span comes out.
W5_MAX_LIVE_ROW = W5_LIVE_NEED - 1
#: The two debts, from the GROW-DEBT-UNPAID lines.
W5_DEBT = {"PP0": 88064, "PP2": 6144}
#: The gap the group never closed.
W5_GAP = W5_LIVE_NEED - min(W5_BACKED.values())  # 4097
RESERVATION = 462163


class _FakeAlloc:
    def __init__(self, size: int):
        self.size = int(size)
        self.free_pages = torch.arange(1, int(size) + 1, dtype=torch.int64)
        self.residency_withheld_slots = 0


class _FakeVmmPool:
    """A VMM-backed pool with a hard ceiling on what it can commit.

    ``budget_rows`` is what makes the "cannot grow" arm real rather than
    hypothetical: PP1 sits on an RTX 3080 with an 18800 MiB budget and PP0 on a
    5090 with 31800, so a floor rank that physically cannot reach the need is
    the expected case on this rig, not an edge one.
    """

    def __init__(self, backed_rows: int, reserved_rows: int, budget_rows=None):
        self.full_pool_backed_rows = int(backed_rows)
        self.reserved = int(reserved_rows)
        self.reserved_backing_rows = int(reserved_rows)
        self.size = int(reserved_rows)
        self.page_size = 1
        self.budget_rows = int(budget_rows) if budget_rows is not None else None
        self.attempts = []

    def runtime_set_backing_rows(self, rows: int) -> None:
        rows = int(rows)
        self.attempts.append(rows)
        if self.budget_rows is not None and rows > self.budget_rows:
            raise MemoryError(
                f"cannot commit {rows} rows, budget is {self.budget_rows}"
            )
        self.full_pool_backed_rows = rows


class _Rank:
    def __init__(self, name: str, rows: int, budget_rows=None):
        from sglang.srt.managers.kv_backing_relief import KvBackingRelief

        self.name = name
        self.pool = _FakeVmmPool(rows, RESERVATION, budget_rows=budget_rows)
        self.alloc = _FakeAlloc(RESERVATION)
        self.relief = KvBackingRelief(
            self.pool,
            allocator=self.alloc,
            live_slots_fn=lambda: torch.empty((0,), dtype=torch.int64),
            bytes_per_row=BYTES_PER_ROW,
            probe=lambda: 8192 * (1 << 20),
            device_index=0,
            buffers=1,
            law_floor_bytes=LAW_FLOOR,
            pool_fn=lambda: self.pool,
        )

    def backed(self) -> int:
        return int(self.relief.backed_rows())

    def exposed(self) -> int:
        return int(self.relief.exposed_rows())

    def pay_debt(self, rows: int) -> None:
        """A deferred #834 grow, paid. Rank-local: pages commit, nothing announces."""
        self.pool.runtime_set_backing_rows(self.pool.full_pool_backed_rows + int(rows))


def _group(budgets=None):
    budgets = budgets or {}
    return {
        name: _Rank(name, rows, budget_rows=budgets.get(name))
        for name, rows in W5_BACKED.items()
    }


def _ballot(ranks, *, need=W5_LIVE_NEED):
    """One seam round, modelling phase_flip_spill.py:2042-2070.

    Both numbers come out of the SAME reduced payload, exactly as
    ``collective_slot_ballot`` decodes them. The point of the test is what the
    callsite does with the second one.
    """
    floor = min(r.backed() for r in ranks.values())
    for r in ranks.values():
        r.relief.note_group_backing_floor(floor)
        note_need = getattr(r.relief, "note_group_live_need", None)
        if callable(note_need):
            note_need(int(need) - 1)  # ballot carries the ROW ID; the rung adds the +1
        # ORDER MATTERS AND MIRRORS THE CALLSITE: publish first, so this round's
        # exposure follows THIS round's verdict, then commit pages for the next
        # one. Growing before publishing would let the same round both grow and
        # announce, which is the raise #839 A forbids.
        r.relief.publish_group_exposure("seam ballot")
        close = getattr(r.relief, "close_floor_need_gap", None)
        if callable(close):
            close()
    return floor


def _ballot_level_only(ranks, *, need=W5_LIVE_NEED):
    """The round-5 ballot: take the floor, publish, and drop the need.

    This is integ/round5's behaviour exactly, and the characterisation tests
    below are written against it on purpose -- they state an arithmetic fact
    about ``min`` that must remain true no matter what the rung does next.
    """
    floor = min(r.backed() for r in ranks.values())
    for r in ranks.values():
        r.relief.note_group_backing_floor(floor)
        note_need = getattr(r.relief, "note_group_live_need", None)
        if callable(note_need):
            note_need(int(need) - 1)  # ballot carries the ROW ID; the rung adds the +1
        r.relief.publish_group_exposure("seam ballot")
    return floor


class TheDebtOfANonFloorRankIsStructurallyUnpayable(unittest.TestCase):
    """Characterisation. GREEN on the base tree -- this is the true arithmetic.

    It is here because it is the fact the window-5 result rests on, and because
    a future "fix" that made this test fail would be papering over the min.
    """

    def test_paying_every_deferred_grow_moves_the_group_level_by_zero(self):
        ranks = _group()
        before = _ballot_level_only(ranks)
        self.assertEqual(before, 126976)
        for name, rows in W5_DEBT.items():
            ranks[name].pay_debt(rows)
        after = _ballot_level_only(ranks)
        self.assertEqual(
            after,
            before,
            "paying the debt of ranks that are not the floor must not move the "
            "level -- if this changes, the min is no longer the min",
        )
        self.assertLess(after, W5_LIVE_NEED, "and the flip still cannot fit")

    def test_only_the_floor_rank_growing_moves_the_level(self):
        ranks = _group()
        _ballot_level_only(ranks)
        ranks["PP1"].pay_debt(W5_GAP)
        self.assertEqual(_ballot_level_only(ranks), W5_LIVE_NEED)


class TheBallotsLiveNeedReachesTheRung(unittest.TestCase):
    """RED on integ/round5: nothing consumes ``verdict["max_live_row"]``."""

    def test_the_rung_can_be_told_the_group_live_need(self):
        ranks = _group()
        rung = ranks["PP1"].relief
        self.assertTrue(
            callable(getattr(rung, "note_group_live_need", None)),
            "the ballot decodes max_live_row in the same payload as "
            "min_backed_rows (kv_backing_relief.py:537-543); the rung must "
            "have somewhere to put it",
        )

    def test_the_floor_rank_reports_the_gap_it_must_close(self):
        ranks = _group()
        _ballot_level_only(ranks)
        self.assertEqual(int(ranks["PP1"].relief.floor_need_gap()), W5_GAP)

    def test_a_rank_that_is_not_the_floor_reports_no_gap(self):
        """A richer rank must not try to close a gap that is not its to close."""
        ranks = _group()
        _ballot_level_only(ranks)
        for name in ("PP0", "PP2"):
            self.assertEqual(
                int(ranks[name].relief.floor_need_gap()),
                0,
                f"{name} backs more than the floor; the gap is PP1's",
            )

    def test_the_gap_is_zero_once_the_floor_covers_the_need(self):
        ranks = _group()
        ranks["PP1"].pay_debt(W5_GAP)
        _ballot_level_only(ranks)
        self.assertEqual(int(ranks["PP1"].relief.floor_need_gap()), 0)


class TheFloorRankClosesTheGap(unittest.TestCase):
    """RED on integ/round5. The group must converge without a cutover."""

    def test_two_ballots_are_enough_to_cover_the_live_set(self):
        """Round 1 grows the floor; round 2's group verdict raises exposure.

        Deliberately TWO rounds, not one: the grow commits pages in round 1 and
        only the NEXT group verdict may announce them. One round would mean the
        raise had happened without a verdict measured after the grow.
        """
        ranks = _group()
        self.assertEqual(_ballot(ranks), 126976)
        self.assertEqual(_ballot(ranks), W5_LIVE_NEED)
        for name, rank in ranks.items():
            self.assertEqual(
                rank.exposed(),
                W5_LIVE_NEED,
                f"{name} must expose exactly the agreed level",
            )

    def test_no_cutover_was_needed(self):
        """The window-5 cycle: level rises only at a cutover, cutover needs the level.

        Nothing in this test performs a flip. If it passes, the cycle is broken.
        """
        ranks = _group()
        _ballot(ranks)
        floor = _ballot(ranks)
        self.assertGreaterEqual(floor, W5_LIVE_NEED)


class TheDangerDirectionIsHeld(unittest.TestCase):
    """These must be GREEN BEFORE AND AFTER. They are the reason for the shape."""

    def test_growing_the_floor_rank_never_raises_its_own_exposure(self):
        """A grow may commit pages. It may NEVER announce them. #839 A.

        The shortcut a future edit will be tempted to take is to raise exposure
        inside the grow, which turns two rounds into one. That is exactly the
        window-4-A defect: a rank-local reading licensing a raise.
        """
        ranks = _group()
        _ballot_level_only(ranks)
        before = ranks["PP1"].exposed()
        ranks["PP1"].pay_debt(W5_GAP)
        self.assertEqual(
            ranks["PP1"].exposed(),
            before,
            "committing pages must not move the exposed id space by itself",
        )

    def test_close_floor_need_gap_itself_never_moves_the_exposed_id_space(self):
        """THE ACTUATOR, not a stand-in for it. Mutant M2 found this hole.

        ``test_growing_the_floor_rank_never_raises_its_own_exposure`` writes to
        the pool directly, so it never executes ``close_floor_need_gap`` and a
        version of that method which announced its own pages survived the whole
        suite. This one drives the real method and reads the real exposure.
        """
        ranks = _group()
        rung = ranks["PP1"].relief
        close = getattr(rung, "close_floor_need_gap", None)
        if not callable(close):
            self.skipTest("pre-fix tree")
        _ballot_level_only(ranks)
        before = ranks["PP1"].exposed()
        grown = int(close())
        self.assertEqual(grown, W5_GAP, "the pages must actually be committed")
        self.assertGreater(ranks["PP1"].backed(), before, "backing must have risen")
        self.assertEqual(
            ranks["PP1"].exposed(),
            before,
            "close_floor_need_gap committed pages AND announced them -- that is "
            "a rank-local reading licensing a raise, the window-4-A defect",
        )

    def test_exposure_never_exceeds_the_group_floor(self):
        ranks = _group()
        for _ in range(4):
            floor = _ballot(ranks)
            for name, rank in ranks.items():
                self.assertLessEqual(
                    rank.exposed(),
                    floor,
                    f"{name} exposed an id above the group floor",
                )

    def test_a_rank_never_exposes_more_than_it_has_backed(self):
        ranks = _group()
        for _ in range(4):
            _ballot(ranks)
            for name, rank in ranks.items():
                self.assertLessEqual(
                    rank.exposed(),
                    rank.backed(),
                    f"{name} exposed an id with no page behind it (#816)",
                )

    def test_a_need_with_no_group_verdict_cannot_raise_anything(self):
        """The need alone is not a licence. Without a floor there is no raise."""
        ranks = _group()
        rung = ranks["PP1"].relief
        note_need = getattr(rung, "note_group_live_need", None)
        if not callable(note_need):
            self.skipTest("pre-fix tree: the need has nowhere to go")
        before = ranks["PP1"].exposed()
        note_need(10 ** 9)
        self.assertEqual(int(rung.publish_group_exposure("seam ballot")), 0)
        self.assertEqual(ranks["PP1"].exposed(), before)


    def test_a_floor_measured_in_another_arena_licenses_no_grow(self):
        """The #839 A rule applies to the NEED as well as to the floor.

        A need is a row id, and a row id from the other layout is not a bigger
        or smaller need -- it is a different quantity. Acting on a cross-arena
        pair is how window 4 A published three levels from one ``min``.
        """
        ranks = _group()
        rung = ranks["PP1"].relief
        note_need = getattr(rung, "note_group_live_need", None)
        if not callable(note_need):
            self.skipTest("pre-fix tree")
        _ballot_level_only(ranks)
        self.assertEqual(int(rung.floor_need_gap()), W5_GAP)
        # Move the arena the backing calls resolve to, exactly as a flip leg
        # does, WITHOUT a new ballot. Neither stamp matches now.
        ranks["PP1"].pool = _FakeVmmPool(W5_BACKED["PP1"], RESERVATION)
        self.assertEqual(
            int(rung.floor_need_gap()),
            0,
            "a floor/need pair measured in the other arena must license nothing",
        )


class AFloorRankThatCannotGrowSaysSoOnce(unittest.TestCase):
    """RED on integ/round5: 153 silent abandons, no named refusal.

    This is the arm that matters on THIS rig. PP1 lives on a 20 GB 3080 with an
    18800 MiB budget; a live set that outgrows what it can commit is the normal
    case, not the edge one. The correct behaviour is one named refusal naming
    the binding rank, its backing, the need and the shortfall -- not silence.
    """

    def test_an_unreachable_need_is_refused_by_name_and_not_retried_silently(self):
        ranks = _group(budgets={"PP1": 128000})
        _ballot(ranks)
        _ballot(ranks)
        refusal = getattr(ranks["PP1"].relief, "floor_need_refusal", None)
        self.assertTrue(
            callable(refusal),
            "a floor rank that cannot fund the group's live set must name it",
        )
        said = refusal()
        self.assertIsNotNone(said, "the refusal must be readable, not only logged")
        self.assertEqual(int(said["binding_rows"]), 126976)
        self.assertEqual(int(said["need"]), W5_LIVE_NEED)
        self.assertEqual(int(said["short"]), W5_GAP)

    def test_the_refusal_is_said_once_and_not_once_per_round(self):
        """Window 5 printed 1368 GROW-DEBT-UNPAID lines and named the wrong ranks.

        An alarm repeated every round is not more informative than one and it
        does crowd the log. The refusal must be idempotent while the situation
        is unchanged.
        """
        ranks = _group(budgets={"PP1": 128000})
        rung = ranks["PP1"].relief
        for _ in range(6):
            _ballot(ranks)
        refusal = getattr(rung, "floor_need_refusal", None)
        if not callable(refusal):
            self.skipTest("pre-fix tree")
        said = refusal()
        self.assertIsNotNone(said)
        attempts = ranks["PP1"].pool.attempts
        self.assertGreaterEqual(len(attempts), 1, "it must have tried at least once")
        self.assertEqual(
            len(set(attempts)),
            1,
            "the same unreachable target must not be re-derived as a new one",
        )

    def test_a_failed_grow_leaves_the_group_exactly_where_it_was(self):
        """A MemoryError mid-grow must not corrupt the level or the exposure."""
        ranks = _group(budgets={"PP1": 128000})
        first = _ballot(ranks)
        second = _ballot(ranks)
        self.assertEqual(second, first)
        for name, rank in ranks.items():
            self.assertLessEqual(rank.exposed(), rank.backed())
            self.assertLessEqual(rank.exposed(), second)


class TheCallsiteActuallyConsumesTheNeed(unittest.TestCase):
    """RED on integ/round5. A rung nobody drives is the W9 failure shape.

    #839's own history is the reason this pin exists: a shipped fix that was
    never reached from production code, with no log line saying so, is exactly
    what window 5 spent two boots and one regen discovering.
    """

    def _spill_source(self) -> str:
        import inspect

        from sglang.srt.managers import phase_flip_spill

        return inspect.getsource(phase_flip_spill)

    def test_the_ballot_callsite_reads_max_live_row(self):
        # assertTrue, not assertIn: assertIn renders the WHOLE module source
        # into the failure message, which buries the one line that matters
        # under 3000 lines of unrelated docstring.
        self.assertTrue(
            'verdict.get("max_live_row")' in self._spill_source(),
            'phase_flip_spill never reads verdict["max_live_row"]',
        )

    def test_the_ballot_callsite_drives_both_halves(self):
        """Pin the CALLS, not the names. Mutant M6 found this hole.

        Checking for the string ``close_floor_need_gap`` passes on a callsite
        that only does ``getattr(..., "close_floor_need_gap", None)`` and then
        never invokes it -- which is precisely the inert-fix shape this ticket
        exists to fix. The bound names are what must be invoked.
        """
        src = self._spill_source()
        for name in ("note_group_live_need", "close_floor_need_gap"):
            self.assertTrue(name in src, f"{name} is never reached from production")
        for call in ("note_need(int(need))", "close_gap()"):
            self.assertTrue(call in src, f"{call} is looked up but never invoked")

    def test_the_grow_is_driven_after_the_publication_not_before(self):
        """Order is the safety property, so the order is pinned, not assumed."""
        src = self._spill_source()
        # Guard the lookups so a pre-fix tree fails with the reason rather than
        # a bare ValueError from str.index.
        self.assertTrue(
            "close_floor_need_gap" in src, "the grow is never driven at all"
        )
        self.assertLess(
            src.index("publish_group_exposure"),
            src.index("close_floor_need_gap"),
            "publishing must precede growing, or a round both grows and "
            "announces -- the raise #839 A forbids",
        )


if __name__ == "__main__":
    unittest.main()
