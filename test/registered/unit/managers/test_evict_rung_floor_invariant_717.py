"""#717: a lowered floor may never fall below the highest LIVE row.

WHAT HAPPENED. The first #717 fix (c4e557963e) taught the rung that "nothing
resident" is not "unknown" -- correct, and still correct. It was reverted
(b7868580a9) because it caused flip crashes: backing shrank to 69,054 rows
while the highest live row was 233,289, and the next access to a row above
the cap was an illegal address.

WHY THE ORIGINAL PINS DID NOT CATCH IT. Every one of them tested
``_evict_floor_rows`` -- the PRICE. None tested ``_shrink_to`` -- the APPLY.
In the pricing domain the fix reads correctly; the crash lives one layer
down, and the suite never went there.

THE TWO FACTS THAT COMBINE. Neither is a bug alone:

  1. "No running requests" is NOT "no live rows". The radix tree's cached
     rows are live and addressable; ``_max_live_row`` reports them, and it
     returns 0 only for a genuinely empty live set. On the specimen the tree
     held rows up to 397,958 with ZERO resident requests.
  2. ``_lower_watermark_to`` REFUSES when ``_resident_ceiling()`` is negative
     and returns 0 -- and ``_shrink_to`` discards that return value. So the
     eviction the low target assumed never ran, and the cap engaged anyway.

The first fix opened PRICING on the nothing-resident branch while EVICTION
still refused on it. The floor followed intention, not completion.

THE INVARIANT PINNED HERE, and it holds no matter what the pricing says:
after the eviction step, the engaged cap is at or above
``_floor_rows(highest live row)``. If the eviction did not deliver the mark
the target assumed, the target must rise to what the live set actually
permits -- or the shrink must not happen. Stated as a rule about ORDER: the
floor follows COMPLETION, not intention.

This is the safety property. The pricing question (#717 proper) is pinned in
``test_evict_rung_nothing_resident_717.py``; it is only safe to reopen
because this file holds the floor underneath it.
"""

import unittest

import torch

from sglang.srt.managers import kv_backing_relief as m

# The crash, verbatim.
CRASH_LIVE_ROW = 233_289
CRASH_CAP = 69_054

MARGIN = 1
RESERVE = 511


class _Cap:
    def __init__(self):
        self.engaged = []
        self.released = 0

    def engage(self, target):
        self.engaged.append(int(target))

    def release(self):
        self.released += 1


class _Pool:
    page_size = 1

    def __init__(self):
        self.backing_calls = []

    def runtime_set_backing_rows(self, target):
        self.backing_calls.append(int(target))
        return int(target)


def _relief(live_max, split, *, evicts_to=None):
    """A relief whose eviction step delivers exactly what the test dictates.

    ``evicts_to`` is the live max AFTER the eviction runs -- None means the
    eviction delivers nothing, which is the production behaviour whenever
    ``_lower_watermark_to`` refuses. Modelling the outcome rather than the
    tree lets the pin state the invariant in the only terms that matter to
    it: what the live set is when the cap engages.
    """
    r = m.KvBackingRelief.__new__(m.KvBackingRelief)
    r._pool = _Pool()
    r._cap = _Cap()
    r._margin_rows = MARGIN
    r._admission_reserve_rows = RESERVE
    r._rows_at_boot = None
    r._exhausted = False
    r._bytes_per_row = 1024
    # #815 STUB DRIFT, and this one needed a MEASUREMENT rather than an
    # argument. 9aec623b2a [#796] added a `self._buffers` read to `_shrink_to`'s
    # "the pool reported a release the driver's free column did not show"
    # diagnostic; this helper builds the relief with `__new__` and never had the
    # field. `_buffers` is real -- assigned unconditionally in __init__ and
    # again on the restore path.
    #
    # 0 IS THE CONSTRUCTOR-FAITHFUL VALUE HERE, not a placeholder: the `buffers`
    # parameter defaults to 0, this file's `_Pool` declares no buffer geometry
    # at all (page_size=1, nothing else), so a real KvBackingRelief built over
    # THIS pool would carry 0. Production treats `_buffers <= 0` as a modelled
    # state, not an invalid one -- it guards the chunk-to-rows conversion with
    # exactly that test.
    #
    # And the choice is not load-bearing, which was checked rather than assumed:
    # the file was run with 0, 1 and 28, all 5 passed in each case. These tests
    # assert on `_cap.engaged` and `_pool.backing_calls`, neither of which this
    # value reaches. Recorded so the next reader does not re-derive it.
    r._buffers = 0
    r._last_live_split = split
    r._tree_cache_fn = lambda: object()
    r.evicted_rows_total = 0
    r.evict_count = 0

    state = {"max": int(live_max)}

    def _live():
        return torch.tensor([state["max"]], dtype=torch.int64)

    _live.last_split = split
    r._live_slots_fn = _live

    def _lower(target):
        if evicts_to is None:
            return 0
        state["max"] = int(evicts_to)
        return 1

    r._lower_watermark_to = _lower
    r._free_bytes = lambda: 0
    r._min_release_rows = lambda: 1
    r._mark_exhausted = lambda target: None
    return r


class TestTheFloorFollowsCompletion(unittest.TestCase):
    """The safety property, stated as behaviour at the apply site."""

    def test_the_crash_shape_does_not_cap_below_the_live_rows(self):
        """THE PIN. Nothing resident, eviction delivers nothing, low target.

        This is the reverted commit's crash reproduced at the layer it
        actually occurred on: 69,054 rows of backing under a highest live row
        of 233,289.
        """
        r = _relief(CRASH_LIVE_ROW, {"req_max": -1, "req_rows": 0})

        r._shrink_to(CRASH_CAP, current=400_000)

        safe = r._floor_rows(CRASH_LIVE_ROW)
        for engaged in r._cap.engaged:
            self.assertGreaterEqual(
                engaged,
                safe,
                f"capped to {engaged} rows with live rows up to "
                f"{CRASH_LIVE_ROW}: every access above the cap is an illegal "
                f"address",
            )
        for asked in r._pool.backing_calls:
            self.assertGreaterEqual(asked, safe)

    def test_an_eviction_that_delivers_permits_the_low_target(self):
        """The invariant must not become 'never shrink'.

        When the eviction really does bring the mark down, the low target is
        legitimate and must go through -- otherwise the rung is dead again,
        which is the defect #717 exists to fix.
        """
        r = _relief(
            CRASH_LIVE_ROW, {"req_max": -1, "req_rows": 0}, evicts_to=CRASH_CAP - 2000
        )

        r._shrink_to(CRASH_CAP, current=400_000)

        self.assertEqual(
            r._cap.engaged,
            [CRASH_CAP],
            "a delivered eviction must let the priced target stand",
        )

    def test_a_partial_eviction_caps_at_what_it_actually_delivered(self):
        """Under-delivery is not refusal, and must not be treated as either
        'proceed as asked' or 'do nothing'. The cap follows the mark the
        eviction actually reached."""
        delivered = 150_000
        r = _relief(CRASH_LIVE_ROW, {"req_max": -1, "req_rows": 0}, evicts_to=delivered)

        r._shrink_to(CRASH_CAP, current=400_000)

        safe = r._floor_rows(delivered)
        self.assertTrue(r._cap.engaged, "a shrink to the safe floor should occur")
        for engaged in r._cap.engaged:
            self.assertGreaterEqual(engaged, safe)

    def test_no_shrink_at_all_when_the_safe_floor_is_not_below_current(self):
        """If the live set permits nothing, the shrink is abandoned rather
        than performed at a token depth: capping to `current` would burn the
        seam's one attempt for zero bytes."""
        r = _relief(CRASH_LIVE_ROW, {"req_max": -1, "req_rows": 0})

        freed = r._shrink_to(CRASH_CAP, current=r._floor_rows(CRASH_LIVE_ROW))

        self.assertEqual(freed, 0)
        self.assertEqual(r._cap.engaged, [], "nothing may be capped here")

    def test_an_unreadable_live_set_shrinks_nothing(self):
        """`_max_live_row` returns -1 when the probe fails. An unknown live
        set is not an empty one, so the apply site must refuse outright --
        the same conservative reading the pricing side takes."""
        r = _relief(CRASH_LIVE_ROW, None)

        def _boom():
            raise RuntimeError("probe failed")

        _boom.last_split = None
        r._live_slots_fn = _boom

        freed = r._shrink_to(CRASH_CAP, current=400_000)

        self.assertEqual(freed, 0)
        self.assertEqual(r._cap.engaged, [])


if __name__ == "__main__":
    unittest.main()
