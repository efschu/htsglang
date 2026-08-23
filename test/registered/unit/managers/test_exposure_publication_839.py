"""#839 A -- exposure is PUBLISHED after the group floor lands, never before.

THE SPECIMEN, boot_window4A_0823_2059, /spinning/evidence-665-f1/
window4A_clamp_divergence_2106/. Three group-uniform rounds, then one round in
which the group stops agreeing, and the last completed flip is two seconds
after it::

    21:04:52 PP1 Capped at 122880    (this rank committed 462848)
    21:04:53 PP2 Capped at 122880    (this rank committed 462848)
    21:04:54 PP0 Capped at 122880    (this rank committed 462848)
    21:05:39 PP1 Capped at 122880    (this rank committed 462848)
    21:05:39 PP2 Capped at 122880    (this rank committed 462848)
    21:05:41 PP0 Capped at 122880    (this rank committed 462848)
    21:06:11 PP1 Capped at 122880    (this rank committed 122880)
    21:06:11 PP2 Capped at 131072    (this rank committed 131072)
    21:06:12 PP0 Capped at 210944    (this rank committed 210944)
    21:06:13     last completed flip

From there every ``tp_to_pp`` abandoned, and the wire-frame refusals name PP0's
diverged ceiling to the row::

    12 x "the union reaches row 210944 and the poorest rank has only 122880"
    10 x 209468, 6 x 201327, 6 x 122959      (40 refusals, 0 flips)

THE ATTRIBUTION, and it is in those numbers rather than in the abandon text.
Read the two columns of each clamp line together. In the first six lines the
rank's own committed count is 462848 and the ceiling is 122880 -- so the
ceiling came from the group floor, and the floor was 122880. In the last three
the ceiling EQUALS each rank's own committed count -- so in that round the
floor did not bind on any of them. Yet the ballot's floor at 21:06:39 is still
122880, named 377 times in the abandon lines. One quantity, two readings, and
they are readings of DIFFERENT LAYOUTS:

    ``KvBackingRelief`` serves two arenas (``_rebind``, kv_backing_relief.py
    :994). The seam ballot reads ``backed_rows()`` -- which rebinds -- and
    hands the group MIN to ``note_group_backing_floor``
    (phase_flip_spill.py:2042 -> kv_backing_relief.py:2828). The exposure
    clamp reads ``_current_rows()`` in whichever arena is active when IT runs
    (kv_backing_relief.py:2892). A row count from the narrow layout and a row
    count from the wide one are not two sizes of the same thing.

    Rounds 1-3: floor narrow (122880), backing wide (462848). ``min`` bound,
    and the group was uniform BY LUCK -- the stale reading happened to be the
    smaller one.
    Round 4: floor wide, backing narrow. The same ``min`` bound NOTHING, each
    rank published its own backing, and the id space parted company.

The publication path, end to end, is::

    phase_flip_spill.grow_kv_backing_local        :1252
      -> KvBackingRelief.recover                  :2666
        -> KvRowCap.release / engage(now)         :2775-2777   <- rank-local
        -> clamp_exposure_to_backing              :2835
          -> group_exposure_ceiling(backed, floor):2892        <- floor, later

The rank-local level is put on the allocator first and the group bound is
applied afterwards, out of a value nobody checked was measured in this arena.

SEGMENT B IS THE NATURAL EXPERIMENT AND IT IS NOT THE PROOF. The same tree
with ``SGLANG_SEAM_SHRINK=1`` -- which defers the grow, so the levelling and
the clamp are adjacent instead of a round apart -- produced ONE clamp level
(118784) over 36 lines and zero "poorest rank" refusals. It differs only in
that ordering, which is why it is cited; it is an A/B on metal, not a bisect,
and the file:line path above is what carries the claim.

THE FIX IS AN ORDERING RULE, not a new collective. Exposure may be LOWERED by
any reading at any time; it may be RAISED only by a group verdict measured in
the arena the raise happens in. A rank-local grow may commit pages whenever it
likes -- what it may not do is announce them.
"""

import unittest

import torch

BYTES_PER_ROW = 32768
LAW_FLOOR = 1024 * 1024 * 1024

#: boot_window4A_0823_2059, the diverged round, verbatim.
W4_NARROW = {"PP1": 122880, "PP2": 131072, "PP0": 210944}
#: The same three ranks' committed count in the OTHER arena, verbatim from the
#: first six clamp lines: identical on every rank.
W4_WIDE = 462848
#: "the allocator could hand out 462163 rows", on all nine lines.
W4_RESERVATION = 462163
#: The floor the ballot was still reporting at 21:06:39, 377 times.
W4_BALLOT_FLOOR = min(W4_NARROW.values())


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


class _Rank:
    """One rank with TWO arenas, which is the whole point of the specimen.

    ``switch_to`` moves the layout the backing calls resolve to, exactly as
    ``_rebind`` does on a flip leg. The id space stays anchored on the pool the
    rung was built with, as ``_reservation_rows`` requires.
    """

    def __init__(self, name: str, narrow_rows: int):
        from sglang.srt.managers.kv_backing_relief import KvBackingRelief

        self.name = name
        self.narrow = _FakeVmmPool(narrow_rows, W4_RESERVATION)
        self.wide = _FakeVmmPool(W4_WIDE, W4_RESERVATION)
        self.active = self.narrow
        self.alloc = _FakeAlloc(W4_RESERVATION)
        self.relief = KvBackingRelief(
            self.narrow,
            allocator=self.alloc,
            live_slots_fn=lambda: torch.empty((0,), dtype=torch.int64),
            bytes_per_row=BYTES_PER_ROW,
            probe=lambda: 8192 * (1 << 20),
            device_index=0,
            buffers=1,
            law_floor_bytes=LAW_FLOOR,
            pool_fn=lambda: self.active,
        )

    def switch_to(self, which: str) -> None:
        self.active = self.narrow if which == "narrow" else self.wide

    def backed(self) -> int:
        """The public reading the ballot uses. It rebinds; that matters."""
        return int(self.relief.backed_rows())

    def exposed(self) -> int:
        return int(self.relief.exposed_rows())

    def recover_tail(self) -> None:
        """``recover``'s publication sequence, kv_backing_relief.py:2772-2835.

        THE RANK-LOCAL LEVEL GOES OUT FIRST. ``recover`` releases the cap and
        re-engages it at ``now`` -- this rank's own freshly grown backing --
        and only then calls ``clamp_exposure_to_backing`` to apply the group
        bound. ``reconcile_to`` is that same release-and-re-engage as a public
        call, so this models the shipped order without reaching into privates.

        Modelling it matters: ``clamp_exposure_to_backing`` alone can only
        LOWER, so a test that called it by itself could never reproduce a rank
        RAISING its exposure past the group -- and raising is the defect.
        """
        backed = self.relief.backed_rows()
        self.relief.reconcile_to(backed)
        self.relief.clamp_exposure_to_backing("after recovery")


def _group(arena: str = "narrow"):
    ranks = {name: _Rank(name, rows) for name, rows in W4_NARROW.items()}
    for rank in ranks.values():
        rank.switch_to(arena)
    return ranks


def _ballot(ranks, arena: str) -> int:
    """One seam round of the rung's reduction, in ``arena``.

    Every rank reports its backing, the group MIN is decoded identically on
    each of them, and each takes it -- which is what
    ``collective_kv_backing_relief`` does at phase_flip_spill.py:2042.
    """
    for rank in ranks.values():
        rank.switch_to(arena)
    floor = min(rank.backed() for rank in ranks.values())
    for rank in ranks.values():
        rank.relief.note_group_backing_floor(floor)
        publish = getattr(rank.relief, "publish_group_exposure", None)
        if publish is not None:
            publish("seam ballot")
    return floor


class TheWindow4ADivergenceIsClosed(unittest.TestCase):
    """The specimen's numbers, replayed through the real actuator.

    RED ON THE BASE TREE: the last assertion sees {122880, 131072, 210944} --
    the three ceilings the boot printed, to the row.
    """

    def test_the_group_agrees_one_ceiling_while_a_grow_is_in_flight(self):
        ranks = _group("narrow")

        # Rounds 1-3 of the specimen. The ballot runs in the narrow arena and
        # decides 122880; the clamp then runs with the WIDE layout active.
        floor = _ballot(ranks, "narrow")
        self.assertEqual(floor, W4_BALLOT_FLOOR)
        for rank in ranks.values():
            rank.switch_to("wide")
            rank.recover_tail()
        uniform = {rank.exposed() for rank in ranks.values()}
        self.assertEqual(
            uniform,
            {W4_BALLOT_FLOOR},
            "rounds 1-3 were uniform on metal and must stay uniform: "
            f"{ {n: r.exposed() for n, r in ranks.items()} }",
        )

        # THE GROW IN FLIGHT. A ballot lands while the wide layout is active,
        # so the floor now counts rows in the arena the ranks are NOT about to
        # clamp in. This is the only thing that changes between round 3 and
        # round 4 of the specimen.
        wide_floor = _ballot(ranks, "wide")
        self.assertEqual(wide_floor, W4_WIDE)

        # Round 4: back on the narrow layout, each rank clamps.
        for rank in ranks.values():
            rank.switch_to("narrow")
            rank.recover_tail()
        levels = {name: rank.exposed() for name, rank in ranks.items()}
        self.assertEqual(
            len(set(levels.values())),
            1,
            "the exposure clamp diverged across the group while a grow was in "
            f"flight -- the window-4 headline, to the row: {levels}",
        )
        # And it is the level the group last agreed to, not one rank's own.
        self.assertEqual(set(levels.values()), {W4_BALLOT_FLOOR})

    def test_no_rank_exposes_the_row_the_wire_frame_refused(self):
        """The 12 refusals name row 210944. Nobody may issue it."""
        ranks = _group("narrow")
        _ballot(ranks, "narrow")
        for rank in ranks.values():
            rank.switch_to("wide")
            rank.recover_tail()
        _ballot(ranks, "wide")
        for rank in ranks.values():
            rank.switch_to("narrow")
            rank.recover_tail()
        widest = max(rank.exposed() for rank in ranks.values())
        poorest = min(W4_NARROW.values())
        self.assertLessEqual(
            widest,
            poorest,
            "a rank exposes ids above the group's poorest backing, which is "
            "the union bound every abandoned tp_to_pp named",
        )
        self.assertNotEqual(widest, W4_NARROW["PP0"])


class TheOrderingRule(unittest.TestCase):
    """Lowered by anything, raised only by an in-arena group verdict."""

    def _rank(self):
        return _Rank("PP0", W4_NARROW["PP0"])

    def test_a_stale_arena_floor_may_still_lower(self):
        """Wrong-arena is not the same as ignored.

        A floor from the other layout can only be wrong toward too LITTLE
        exposure when it is the smaller number, and too little exposure is a
        capacity loss rather than an id a peer cannot map. Rounds 1-3 of the
        specimen depend on this: the floor was narrow, the backing wide, and
        the group stayed level because the stale floor bound.
        """
        rank = self._rank()
        rank.relief.note_group_backing_floor(W4_BALLOT_FLOOR)
        rank.switch_to("wide")
        rank.recover_tail()
        self.assertEqual(rank.exposed(), W4_BALLOT_FLOOR)

    def test_a_stale_arena_floor_may_never_raise(self):
        """The defect direction, as its own pin."""
        rank = self._rank()
        # Agree a level in the narrow arena.
        rank.relief.note_group_backing_floor(W4_BALLOT_FLOOR)
        rank.relief.publish_group_exposure("seam ballot")
        self.assertEqual(rank.exposed(), W4_BALLOT_FLOOR)
        # A verdict measured in the wide arena arrives. It is a bigger number
        # than anything the narrow layout holds. The rebind is the ballot's
        # own ``backed_rows()`` call, which is what decides the arena the floor
        # is measured in.
        rank.switch_to("wide")
        rank.backed()
        rank.relief.note_group_backing_floor(W4_WIDE)
        # Back to narrow, and clamp. The wide floor must not license a raise.
        rank.switch_to("narrow")
        rank.recover_tail()
        self.assertEqual(
            rank.exposed(),
            W4_BALLOT_FLOOR,
            "a floor measured in the other arena raised this rank's exposure",
        )

    def test_an_in_arena_verdict_does_raise(self):
        """The rule must not become a one-way ratchet -- that is #814.

        This is the same call that pays #834's deferred grow, so a fix for the
        divergence that froze exposure would simply move the failure.
        """
        rank = self._rank()
        rank.relief.note_group_backing_floor(W4_BALLOT_FLOOR)
        rank.relief.publish_group_exposure("seam ballot")
        self.assertEqual(rank.exposed(), W4_BALLOT_FLOOR)
        # The group's poorest rank grows; the next ballot in the SAME arena
        # agrees a higher level.
        raised = min(W4_NARROW["PP2"], W4_NARROW["PP0"])
        rank.relief.note_group_backing_floor(raised)
        moved = rank.relief.publish_group_exposure("seam ballot")
        self.assertEqual(rank.exposed(), raised)
        self.assertEqual(moved, raised - W4_BALLOT_FLOOR)
        self.assertGreater(moved, 0)

    def test_without_any_verdict_the_816_behaviour_is_unchanged(self):
        """Single-rank shapes, stub rungs and hermetic tests keep #816.

        Guessing a floor where no collective exists strands rows for nothing.
        """
        rank = self._rank()
        rank.recover_tail()
        self.assertEqual(rank.exposed(), W4_NARROW["PP0"])

    def test_publishing_commits_no_pages(self):
        """It is an ID decision. The dial must not be touched, in either
        direction -- a publication that allocated would be the OOM
        ``cap_proposal`` records from the first metal boot of the agreement."""
        rank = self._rank()
        rank.relief.note_group_backing_floor(W4_BALLOT_FLOOR)
        rank.relief.publish_group_exposure("seam ballot")
        rank.relief.note_group_backing_floor(W4_NARROW["PP0"])
        rank.relief.publish_group_exposure("seam ballot")
        self.assertEqual(rank.narrow.attempts, [])
        self.assertEqual(rank.narrow.full_pool_backed_rows, W4_NARROW["PP0"])

    def test_an_abstain_sentinel_publishes_nothing(self):
        """A peer that failed to report must not withdraw this id space."""
        rank = self._rank()
        before = rank.exposed()
        rank.relief.note_group_backing_floor(-1)
        self.assertEqual(rank.relief.publish_group_exposure("seam ballot"), 0)
        self.assertEqual(rank.exposed(), before)


class TheFloorCarriesItsArena(unittest.TestCase):
    """The stamp is what makes the comparison legal. Pin it directly."""

    def test_note_records_which_arena_measured_it(self):
        rank = _Rank("PP0", W4_NARROW["PP0"])
        rank.relief.backed_rows()
        rank.relief.note_group_backing_floor(W4_BALLOT_FLOOR)
        narrow_stamp = rank.relief._group_floor_arena
        rank.switch_to("wide")
        rank.relief.backed_rows()
        rank.relief.note_group_backing_floor(W4_WIDE)
        self.assertNotEqual(
            narrow_stamp,
            rank.relief._group_floor_arena,
            "two floors measured in two arenas carry the same stamp, so the "
            "clamp cannot tell a comparable floor from an incomparable one",
        )


class ThePublicationIsWired(unittest.TestCase):
    """Wired, not merely written -- the standing 'inert fix' failure.

    #833 shipped ``note_group_backing_floor`` and the boot proved it was
    reached; what nothing proved was that anything ACTED on the value in the
    same round. Both halves are pinned here at their call site.
    """

    def test_the_rung_publishes_after_taking_the_floor(self):
        import inspect

        from sglang.srt.managers import phase_flip_spill

        src = inspect.getsource(phase_flip_spill.collective_kv_backing_relief)
        self.assertIn(
            "publish_group_exposure",
            src,
            "the rung records the group floor and never publishes it",
        )
        self.assertIn("note_group_backing_floor", src)
        self.assertLess(
            src.index("note_group_backing_floor"),
            src.index("publish_group_exposure"),
            "the publication must follow the floor it publishes",
        )

    def test_the_relief_exposes_the_entry_point(self):
        rank = _Rank("PP0", W4_NARROW["PP0"])
        self.assertTrue(callable(getattr(rank.relief, "publish_group_exposure")))


if __name__ == "__main__":
    unittest.main()
