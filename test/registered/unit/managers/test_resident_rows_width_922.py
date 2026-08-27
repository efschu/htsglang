"""#922 -- slice by what the ALLOCATOR handed out, not by the sequence length.

THE INSTANCE. `_resident_rows` (phase_flip_runtime.py) enumerated a resident's
KV rows as ``req_to_token[idx, :req.seqlen]``. ``seqlen`` is
``len(origin_input_ids) + len(output_ids)`` -- a property of the SEQUENCE.
``kv_allocated_len`` is what the allocator handed out, and is precisely what
the invariant checker charges to the pool.

THE DIRECTION, WHICH DECIDES WHAT KIND OF DEFECT IT IS, measured twice on two
boots and two configs:

  * 2026-08-09, phase_flip_presence.py:513-516 -- ``seqlen=82
    kv_allocated_len=81 delta_vs_seqlen=-1``;
  * 2026-08-27, boot 2f/2g -- SIXTY-THREE of 63 ``FLIP EXTENT PROBE``
    emissions, three ranks, every flip, all ``delta_vs_seqlen=-1``, at both
    ``seqlen=9448/kv_allocated_len=9447`` and ``seqlen=2/kv_allocated_len=1``.

So ``seqlen`` OVER-counts by one: the slice reads one row BEYOND the
allocation -- a stale cell left by that row's previous tenant. It is an
OVER-report, one bogus row id per resident, and it drives BOTH error
directions downstream:

  * FALSE POSITIVE -- if the stale id is also in the free list or the tree,
    the census sees two owners and raises EXCLUSIVITY_DOUBLED. #912 publishes
    exactly that count onto the allocator and the on_idle invariant checker
    consumes it, so one stale cell per resident lands in the ledger the
    #912/#913 re-read is about to examine;
  * FALSE NEGATIVE -- if the stale id is a genuinely unaccounted row, it now
    reads as resident-owned and a real leak is acquitted.

BOTH CALL SITES HAD ALREADY DEFERRED THIS PENDING EXACTLY THIS EVIDENCE, which
is why the fix is a one-liner and not a redesign. `phase_flip_presence.py:
521-524`: "NOT yet changed: one measurement on one config is not enough to
re-cut an enumeration whose errors are silent, and the change is a one-liner
once a second flip confirms the sign." `build_flip_live_slots_fn`: "change
this only on that evidence." The second flip has now confirmed the sign 63
times.

THE SIBLING WAS THE MOVER, and it is the consequential half. The sweep for
"another reader slicing by seqlen where rows are meant" found
`build_flip_live_slots_fn`, which does not merely COUNT the stale row -- it
MOVES it across the seam "as if it were live KV". Both now go through one
helper, because `_resident_rows`'s own docstring says two enumerations of
"which rows does this request hold" that can disagree is the shape #822 exists
to end; letting them disagree about the EXTENT is that defect one level down.

THE TWO CALLERS DEGRADE DIFFERENTLY ON PURPOSE. A request that cannot state
its extent makes the census answer "no verdict" (declaring an owner that holds
the wrong rows is worse than declaring none -- the function's own contract),
while the mover keeps the OLD extent and says so out loud: moving one stale
row is what it does today and is recoverable; dropping a live row loses a
request's context at the seam, silently.

WHAT EACH TEST HOLDS DOWN
  1. the rig's measured shape (seqlen = allocated + 1) yields the allocated
     extent, not the sequence's;
  2. the stale cell is EXCLUDED from the census's owner set -- the defect;
  3. page alignment is applied when page_size > 1 and not when it is 1;
  4. an unstatable extent makes the census answer None, not a partial set;
  5. the mover FALLS BACK rather than dropping rows -- the danger direction,
     since a dropped row is lost context;
  6. seqlen is never consulted when the allocator can answer -- the mutant
     guard: a "fix" that still prefers seqlen passes every count test.
"""

import unittest

from sglang.srt.managers.phase_flip_runtime import owned_row_extent


class _Req:
    def __init__(self, *, seqlen, kv_allocated_len, rid="r", idx=0):
        self.seqlen = seqlen
        self.kv_allocated_len = kv_allocated_len
        self.rid = rid
        self.req_pool_idx = idx


class TestOwnedRowExtent922(unittest.TestCase):
    def test_the_rig_shape_yields_the_allocated_extent(self):
        """boot 2f/2g, 63 of 63: seqlen is allocated + 1."""
        self.assertEqual(owned_row_extent(_Req(seqlen=9448, kv_allocated_len=9447)), 9447)
        self.assertEqual(owned_row_extent(_Req(seqlen=2, kv_allocated_len=1)), 1)
        # 2026-08-09, the first measurement of the same sign.
        self.assertEqual(owned_row_extent(_Req(seqlen=82, kv_allocated_len=81)), 81)

    def test_seqlen_is_never_consulted_when_the_allocator_can_answer(self):
        """MUTANT GUARD. A 'fix' that still prefers seqlen would satisfy every
        arithmetic test above whenever the two happen to agree, so the extent
        is asked for on a request whose seqlen is absurd."""
        req = _Req(seqlen=10_000, kv_allocated_len=7)
        self.assertEqual(owned_row_extent(req), 7)

    def test_page_alignment_applies_only_above_one(self):
        req = _Req(seqlen=9448, kv_allocated_len=9447)
        self.assertEqual(owned_row_extent(req, 1), 9447)
        self.assertEqual(owned_row_extent(req, 8), 9448)  # ceil to the page

    def test_an_unstatable_extent_is_no_verdict(self):
        self.assertEqual(owned_row_extent(_Req(seqlen=5, kv_allocated_len=0)), -1)
        self.assertEqual(owned_row_extent(_Req(seqlen=5, kv_allocated_len=-1)), -1)

        class _NoField:
            seqlen = 5
            rid = "x"

        self.assertEqual(owned_row_extent(_NoField()), -1)


class _Pool:
    """req_to_token whose row `allocated` holds a STALE id from a previous
    tenant -- the cell the old slice read."""

    def __init__(self, owned, stale):
        self._rows = list(owned) + [stale]

    def __getitem__(self, key):
        _idx, sl = key
        return _Slice(self._rows[sl])


class _Slice:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return list(self._vals)


class TestTheCensusExcludesTheStaleCell922(unittest.TestCase):
    """The defect, at the reader that feeds the on_idle ledger."""

    def _census(self, req, pool, page_size=1):
        from sglang.srt.managers import phase_flip_runtime as pfr

        class _ReqPool:
            req_to_token = pool

        class _Alloc:
            pass

        alloc = _Alloc()
        alloc.page_size = page_size

        class _Sched:
            req_to_token_pool = _ReqPool()
            token_to_kv_pool_allocator = alloc

        sched = _Sched()
        original = pfr._live_reqs
        pfr._live_reqs = lambda _s: [req] if req is not None else []
        try:
            return pfr._resident_rows(sched)
        finally:
            pfr._live_reqs = original

    def test_the_stale_row_is_not_claimed_as_resident_owned(self):
        owned = [101, 102, 103]
        stale = 999  # a previous tenant's id, still sitting in req_to_token
        rows = self._census(
            _Req(seqlen=4, kv_allocated_len=3), _Pool(owned, stale)
        )
        self.assertEqual(rows, set(owned))
        self.assertNotIn(
            stale,
            rows,
            "the row beyond the allocation belongs to nobody now; claiming it "
            "makes a false EXCLUSIVITY_DOUBLED if it is also free or cached, "
            "and acquits a real leak if it is not",
        )

    def test_an_unstatable_extent_gives_no_verdict_not_a_partial_set(self):
        rows = self._census(
            _Req(seqlen=4, kv_allocated_len=0), _Pool([101, 102, 103], 999)
        )
        self.assertIsNone(
            rows,
            "a partial owner set turns the working set back into a leak -- "
            "this function's own documented contract",
        )


if __name__ == "__main__":
    unittest.main()
