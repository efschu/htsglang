"""#839-METAL v2 -- every exit of the floor-need path is NAMED, and the clamp is real.

THE SPECIMEN IS WINDOW 6, integ/round6 @ 241e7ac385, and it is a specimen of
MY OWN FIX DOING NOTHING (/spinning/gpu-arb/WINDOW6-RESULT.md)::

    tp_to_pp completions                          0   out of 570 flip arms
                                                      (114 of them forced by RPC)
    pool too small for the live set             567
    [#839] exposure published (seam ballot)       0
    GROW-DEBT-UNPAID                           5584
    [#839-METAL] GROUP FLOOR CANNOT FUND ...      0   <- neither branch fired
    floor rank backing                       126976   <- never moved, 32.8 min

    backs 126976 / 133120 / 215040,  live set needs 131073,  agreed level 126976

NEITHER BRANCH FIRED. No grow AND no named refusal, though those two were meant
to be exhaustive. The window could establish only that the callsite was reached
-- `poorest rank has only` printed, and that value is set inside the same
`verdict is not None` guard -- and then it ran out of things the log could say.

WHY IT COULD NOT SAY MORE. v1's `floor_need_gap` had FIVE bare `return 0`, and
`close_floor_need_gap` had a sixth path that returned 0 after a SUCCESSFUL
commit. "No group verdict yet", "this rank is not the floor", "the group already
fits", "the arenas disagree" and "the pool clamped" were one indistinguishable
zero.

THIS IS THE THIRD INSTANCE OF ONE FORM ON THIS TREE:
  1. WEDGE-RECOVERY 2026-08-22 -- six exits returning None for six causes while
     the log line asserted one of them;
  2. `publish_group_exposure` -- five exits, one `if moved:` line, which made
     "0 seam-ballot publications" read as "unreachable path" when it had in fact
     run on all 153 arms and declined;
  3. this path -- which I built AFTER filing (2) as a defect.

THE ROOT v2 FIXES, and it is the sixth exit: **a setter that returns without
raising is a claim, not evidence.** If `runtime_set_backing_rows` clamps to the
rank's budget rather than raising, v1 computed `grown = 0`, recorded no refusal,
logged nothing, and returned 0 -- byte-identical to "there was no gap". So the
single outcome an operator most needs ("this rank cannot fund the group's live
set") was precisely the outcome that was silent. `AClampingPoolIsNamed` below is
that path, reproduced exactly.

AND THE OFF-BY-ONE. `max_live_row` is the highest live ROW ID; the span a union
must cover is that plus one, which is how the abandon path itself reads it
(`span = int(ballot.get("max_live_row", -1)) + 1`, phase_flip_runtime.py:8754).
v1 compared the raw row id against a row COUNT and so asked for one row less
than the flip needs.
"""

import unittest

import torch

BYTES_PER_ROW = 32768
LAW_FLOOR = 1024 * 1024 * 1024

#: Window 6 / window 5, identical to the row on all three boots.
W6_BACKED = {"PP0": 215040, "PP1": 126976, "PP2": 133120}
#: What the ballot carries: the highest live ROW ID.
W6_MAX_LIVE_ROW = 131072
#: What the flip actually needs: the SPAN, one more than the row id.
W6_LIVE_SPAN = W6_MAX_LIVE_ROW + 1  # 131073, as the abandon line prints it
W6_FLOOR = min(W6_BACKED.values())  # 126976
W6_GAP = W6_LIVE_SPAN - W6_FLOOR  # 4097
#: Flip arms on the window-6 boot that produced zero completions.
W6_ARMS = 570
RESERVATION = 462163


class _FakeAlloc:
    def __init__(self, size):
        self.size = int(size)
        self.free_pages = torch.arange(1, int(size) + 1, dtype=torch.int64)
        self.residency_withheld_slots = 0


class _Pool:
    """A pool that can grow, refuse loudly, or CLAMP SILENTLY.

    ``clamp_at`` is the window-6 root in one parameter: the setter accepts the
    call, raises nothing, and simply does not reach the target. That is what a
    budget-bounded VMM pool does, and v1 read it as success.
    """

    def __init__(self, backed_rows, reserved_rows, clamp_at=None, raise_at=None):
        self.full_pool_backed_rows = int(backed_rows)
        self.reserved = int(reserved_rows)
        self.reserved_backing_rows = int(reserved_rows)
        self.size = int(reserved_rows)
        self.page_size = 1
        self.clamp_at = clamp_at
        self.raise_at = raise_at
        self.attempts = []

    def runtime_set_backing_rows(self, rows):
        rows = int(rows)
        self.attempts.append(rows)
        if self.raise_at is not None and rows > self.raise_at:
            raise MemoryError(f"cannot commit {rows} rows")
        if self.clamp_at is not None and rows > self.clamp_at:
            # NO EXCEPTION. This is the whole defect.
            self.full_pool_backed_rows = int(self.clamp_at)
            return
        self.full_pool_backed_rows = rows


class _Rank:
    def __init__(self, name, rows, clamp_at=None, raise_at=None):
        from sglang.srt.managers.kv_backing_relief import KvBackingRelief

        self.name = name
        self.pool = _Pool(rows, RESERVATION, clamp_at=clamp_at, raise_at=raise_at)
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

    def backed(self):
        return int(self.relief.backed_rows())

    def exposed(self):
        return int(self.relief.exposed_rows())


def _group(**kw):
    return {n: _Rank(n, rows, **kw) for n, rows in W6_BACKED.items()}


def _ballot(ranks, *, row=W6_MAX_LIVE_ROW, close=True):
    """One seam round in the REAL callsite order (phase_flip_spill.py:2042-2110).

    note floor -> publish -> note need -> close gap. The need is handed over as
    the ballot's RAW ``max_live_row``, because that is what the callsite reads;
    turning it into a span is the rung's job and is one of the things under test.
    """
    floor = min(r.backed() for r in ranks.values())
    for r in ranks.values():
        r.relief.note_group_backing_floor(floor)
        r.relief.publish_group_exposure("seam ballot")
        note = getattr(r.relief, "note_group_live_need", None)
        if callable(note):
            note(int(row))
        if close:
            fn = getattr(r.relief, "close_floor_need_gap", None)
            if callable(fn):
                fn()
    return floor


def _exits(rank):
    return rank.relief.floor_need_exits()


class TheSpanIsARowCountNotARowId(unittest.TestCase):
    """RED on round6: v1 stores the raw row id."""

    def test_the_need_is_max_live_row_plus_one(self):
        ranks = _group()
        rung = ranks["PP1"].relief
        rung.note_group_live_need(W6_MAX_LIVE_ROW)
        self.assertEqual(int(rung._group_live_need), W6_LIVE_SPAN)

    def test_the_gap_is_4097_not_4096(self):
        ranks = _group()
        _ballot(ranks, close=False)
        self.assertEqual(int(ranks["PP1"].relief.floor_need_gap()), W6_GAP)
        self.assertEqual(W6_GAP, 4097)


class EveryExitIsNamed(unittest.TestCase):
    """RED on round6: floor_need_verdict / floor_need_exits do not exist."""

    def test_the_api_exists(self):
        rung = _group()["PP1"].relief
        for name in ("floor_need_verdict", "floor_need_exits"):
            self.assertTrue(callable(getattr(rung, name, None)), f"missing {name}")

    def test_no_group_verdict_is_named(self):
        from sglang.srt.managers import kv_backing_relief as K

        rung = _group()["PP1"].relief
        gap, reason = rung.floor_need_verdict()
        self.assertEqual(gap, 0)
        self.assertEqual(reason, K.FLOOR_NEED_NO_GROUP_VERDICT)
        self.assertEqual(_exits(_group()["PP1"]) or {}, {})  # fresh rank, no exits yet

    def test_the_abstain_sentinel_lands_on_no_group_verdict_not_a_bare_zero(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _group()
        rung = ranks["PP1"].relief
        rung.note_group_backing_floor(W6_FLOOR)
        rung.note_group_live_need(-1)  # the abstain sentinel / truncated payload
        self.assertEqual(rung.floor_need_verdict()[1], K.FLOOR_NEED_NO_GROUP_VERDICT)

    def test_group_fits_is_named(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _group()
        rung = ranks["PP1"].relief
        rung.note_group_backing_floor(W6_FLOOR)
        rung.note_group_live_need(W6_FLOOR - 10)
        self.assertEqual(rung.floor_need_verdict()[1], K.FLOOR_NEED_GROUP_FITS)

    def test_not_the_floor_is_named(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _group()
        _ballot(ranks, close=False)
        for name in ("PP0", "PP2"):
            gap, reason = ranks[name].relief.floor_need_verdict()
            self.assertEqual(gap, 0)
            self.assertEqual(reason, K.FLOOR_NEED_NOT_THE_FLOOR)

    def test_gap_is_named(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _group()
        _ballot(ranks, close=False)
        gap, reason = ranks["PP1"].relief.floor_need_verdict()
        self.assertEqual((gap, reason), (W6_GAP, K.FLOOR_NEED_GAP))

    def test_stale_arena_is_named(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _group()
        _ballot(ranks, close=False)
        ranks["PP1"].pool = _Pool(W6_BACKED["PP1"], RESERVATION)
        self.assertEqual(
            ranks["PP1"].relief.floor_need_verdict()[1], K.FLOOR_NEED_STALE_ARENA
        )

    def test_the_census_counts_every_exit_it_took(self):
        ranks = _group()
        for _ in range(3):
            _ballot(ranks, close=False)
            ranks["PP1"].relief.floor_need_verdict()
        census = _exits(ranks["PP1"])
        self.assertTrue(census, "the census must not be empty after three rounds")
        self.assertEqual(sum(census.values()), 3)


class AClampingPoolIsNamed(unittest.TestCase):
    """THE WINDOW-6 ROOT. RED on round6, where this path is entirely silent."""

    def _clamped(self):
        # PP1 may commit only 1024 of the 4097 rows it needs.
        ranks = _group()
        ranks["PP1"] = _Rank("PP1", W6_BACKED["PP1"], clamp_at=W6_FLOOR + 1024)
        return ranks

    def test_a_clamp_reports_the_TRUE_growth_and_still_refuses(self):
        """Partial progress is reported as the number it is, WITH the refusal.

        The defect is the silence, not the partiality: under-reporting 1024
        genuinely committed rows as 0 would be a second lie in the other
        direction.
        """
        ranks = self._clamped()
        _ballot(ranks, close=False)
        grown = int(ranks["PP1"].relief.close_floor_need_gap())
        self.assertEqual(grown, 1024, "the rows actually committed")
        self.assertIsNotNone(
            ranks["PP1"].relief.floor_need_refusal(), "and the shortfall is named"
        )
        self.assertTrue(
            ranks["PP1"].pool.attempts, "it must have actually attempted the commit"
        )

    def test_the_clamp_has_its_own_exit_name(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = self._clamped()
        _ballot(ranks, close=False)
        ranks["PP1"].relief.close_floor_need_gap()
        self.assertIn(K.FLOOR_NEED_COMMIT_CLAMPED, _exits(ranks["PP1"]))

    def test_the_clamp_produces_a_NAMED_REFUSAL_with_the_numbers(self):
        ranks = self._clamped()
        _ballot(ranks, close=False)
        ranks["PP1"].relief.close_floor_need_gap()
        said = ranks["PP1"].relief.floor_need_refusal()
        self.assertIsNotNone(
            said,
            "a pool that clamps must produce the same named refusal a pool that "
            "raises does -- window 6 produced NEITHER",
        )
        self.assertEqual(int(said["binding_rows"]), W6_FLOOR)
        self.assertEqual(int(said["need"]), W6_LIVE_SPAN)
        self.assertEqual(int(said["short"]), W6_GAP)
        self.assertIn("CLAMPED", said["why"])

    def test_the_two_failure_modes_are_DISTINGUISHABLE(self):
        """A raise and a clamp must not land on the same name.

        This is the whole lesson: window 6 could not tell them apart, or tell
        either from 'there was no gap'.
        """
        from sglang.srt.managers import kv_backing_relief as K

        clamp = self._clamped()
        _ballot(clamp, close=False)
        clamp["PP1"].relief.close_floor_need_gap()

        raiser = _group()
        raiser["PP1"] = _Rank("PP1", W6_BACKED["PP1"], raise_at=W6_FLOOR + 1024)
        _ballot(raiser, close=False)
        raiser["PP1"].relief.close_floor_need_gap()

        self.assertIn(K.FLOOR_NEED_COMMIT_CLAMPED, _exits(clamp["PP1"]))
        self.assertNotIn(K.FLOOR_NEED_COMMIT_RAISED, _exits(clamp["PP1"]))
        self.assertIn(K.FLOOR_NEED_COMMIT_RAISED, _exits(raiser["PP1"]))
        self.assertNotIn(K.FLOOR_NEED_COMMIT_CLAMPED, _exits(raiser["PP1"]))

    def test_neither_branch_silent_is_now_impossible(self):
        """Window 6's ACTUAL signature, and the fixture matters.

        On metal PP1 grew by ZERO -- its backing never moved off 126976 in 32.8
        minutes. v1 then computed ``grown = reached - floor = 0``, recorded no
        refusal and logged nothing: BOTH branches silent. A clamp that allows
        partial growth does NOT reproduce that, because v1 returns a positive
        number there and the test would pass on the broken tree -- which is
        exactly what an earlier draft of this test did.

        So the fixture is clamp_at == floor: the commit is accepted, nothing
        moves, and the rank must still say so.
        """
        for kw in (
            {"clamp_at": W6_FLOOR},           # the metal case: zero growth
            {"clamp_at": W6_FLOOR + 1024},    # partial growth
            {"raise_at": W6_FLOOR + 1024},    # loud failure
        ):
            ranks = _group()
            ranks["PP1"] = _Rank("PP1", W6_BACKED["PP1"], **kw)
            _ballot(ranks, close=False)
            grown = int(ranks["PP1"].relief.close_floor_need_gap())
            refusal = ranks["PP1"].relief.floor_need_refusal()
            self.assertIsNotNone(
                refusal,
                f"target missed for {kw} and NOTHING was refused -- this IS the "
                f"window-6 bug (grown={grown})",
            )
            self.assertEqual(int(refusal["need"]), W6_LIVE_SPAN)

    def test_the_metal_case_zero_growth_is_named_and_refused(self):
        """clamp_at == floor: the commit is accepted and the pool does not move."""
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _group()
        ranks["PP1"] = _Rank("PP1", W6_BACKED["PP1"], clamp_at=W6_FLOOR)
        _ballot(ranks, close=False)
        grown = int(ranks["PP1"].relief.close_floor_need_gap())
        self.assertEqual(grown, 0, "nothing was committed")
        self.assertIn(K.FLOOR_NEED_COMMIT_CLAMPED, _exits(ranks["PP1"]))
        said = ranks["PP1"].relief.floor_need_refusal()
        self.assertIsNotNone(said, "and it must NOT be silent -- window 6 was")
        self.assertEqual(int(said["short"]), W6_GAP)
        self.assertEqual(int(ranks["PP1"].backed()), W6_FLOOR)


class TheHealthyPathStillConverges(unittest.TestCase):
    def test_an_unclamped_floor_rank_grows_and_the_group_converges(self):
        ranks = _group()
        self.assertEqual(_ballot(ranks), W6_FLOOR)
        self.assertEqual(_ballot(ranks), W6_LIVE_SPAN)
        for name, r in ranks.items():
            self.assertEqual(r.exposed(), W6_LIVE_SPAN, f"{name}")

    def test_the_grow_exit_is_named_too(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _group()
        _ballot(ranks, close=False)
        self.assertEqual(int(ranks["PP1"].relief.close_floor_need_gap()), W6_GAP)
        self.assertIn(K.FLOOR_NEED_GROWN, _exits(ranks["PP1"]))


class TheDangerDirectionIsHeld(unittest.TestCase):
    """Unchanged from v1 and must stay green: a grow may commit, never announce."""

    def test_close_floor_need_gap_never_moves_the_exposed_id_space(self):
        ranks = _group()
        _ballot(ranks, close=False)
        before = ranks["PP1"].exposed()
        grown = int(ranks["PP1"].relief.close_floor_need_gap())
        self.assertEqual(grown, W6_GAP)
        self.assertGreater(ranks["PP1"].backed(), before)
        self.assertEqual(
            ranks["PP1"].exposed(), before, "committing pages must not announce them"
        )

    def test_exposure_never_exceeds_the_group_floor_or_the_local_backing(self):
        ranks = _group()
        for _ in range(4):
            floor = _ballot(ranks)
            for name, r in ranks.items():
                self.assertLessEqual(r.exposed(), floor, name)
                self.assertLessEqual(r.exposed(), r.backed(), name)

    def test_a_clamped_group_never_raises_exposure_on_anyone(self):
        ranks = _group()
        ranks["PP1"] = _Rank("PP1", W6_BACKED["PP1"], clamp_at=W6_FLOOR + 1024)
        cap = W6_FLOOR + 1024
        for _ in range(4):
            floor = _ballot(ranks)
            # The floor rises TO the clamp and never past it: partial growth is
            # real, and 128000 is still short of the 131073 the flip needs.
            self.assertLessEqual(floor, cap, "the floor cannot rise past a clamp")
            self.assertLess(floor, W6_LIVE_SPAN, "and the group still does not fit")
            for name, r in ranks.items():
                self.assertLessEqual(r.exposed(), floor, name)


class TheExitSetIsExhaustive(unittest.TestCase):
    """A name nobody can reach is a name that lies about coverage."""

    def test_every_declared_exit_is_reachable_by_this_suite(self):
        from sglang.srt.managers import kv_backing_relief as K

        seen = set()
        # healthy: GAP + GROWN + NOT-THE-FLOOR (+ GROUP-FITS once converged)
        ranks = _group()
        for _ in range(3):
            _ballot(ranks)
        for r in ranks.values():
            seen |= set(_exits(r))
        # clamp + raise + no-gap
        for kw in ({"clamp_at": W6_FLOOR + 1024}, {"raise_at": W6_FLOOR + 1024}):
            g = _group()
            g["PP1"] = _Rank("PP1", W6_BACKED["PP1"], **kw)
            _ballot(g, close=False)
            g["PP1"].relief.close_floor_need_gap()
            g["PP0"].relief.close_floor_need_gap()  # NO-GAP via NOT-THE-FLOOR
            seen |= set(_exits(g["PP1"])) | set(_exits(g["PP0"]))
        # no group verdict
        fresh = _group()["PP1"]
        fresh.relief.floor_need_verdict()
        seen |= set(_exits(fresh))
        # stale arena
        st = _group()
        _ballot(st, close=False)
        st["PP1"].pool = _Pool(W6_BACKED["PP1"], RESERVATION)
        st["PP1"].relief.floor_need_verdict()
        seen |= set(_exits(st["PP1"]))
        # pool cannot grow
        nogrow = _group()
        _ballot(nogrow, close=False)
        del nogrow["PP1"].pool.__class__.runtime_set_backing_rows
        try:
            nogrow["PP1"].relief.close_floor_need_gap()
            seen |= set(_exits(nogrow["PP1"]))
        finally:
            _Pool.runtime_set_backing_rows = _Pool.__dict__.get(
                "runtime_set_backing_rows", None
            ) or (lambda self, rows: None)

        missing = set(K.FLOOR_NEED_EXITS) - seen
        self.assertEqual(
            missing, set(), f"declared but unreachable exit names: {sorted(missing)}"
        )


if __name__ == "__main__":
    unittest.main()
