"""#848 second half -- the dial may not expose a size the arena can never back.

WHAT THE FIRST HALF LEFT OPEN. ``9f32666687`` made the floor-need actuator ask
the arena what it can hold before proposing a grow target, turning window 7's
crash into a named ``RESERVATION-CAPPED`` refusal. Its own docstring says what
it deliberately did not do: "The reservation being too small in the first place
is a boot-time sizing defect and is a separate posting." This is that posting --
and the diagnosis it inherited was wrong in the way that matters.

WINDOW 7 READ THE SPECIMEN AS "PP1's pool is RESERVED at 125052 and is already
BACKING 126976 -- its backing is 1924 rows ABOVE its own reservation ceiling",
and concluded the rank-memory split had to hand PP1 more VRAM
(WINDOW7-RESULT.md, "What the next window needs", item 1). Backing cannot
exceed its own VA reservation: ``_check_final`` refuses every such grow
(kv_vmm_backing.py:1269-1272). The two numbers do not describe one quantity.

``BACKING-DIAL call`` carries all of them on one line, and the inversion is on
EVERY rank of that boot, not only on PP1::

    request=126977 prev_size=124704 uniform_backed_rows=126976
        reserved_backing_rows=125052 store_bound_rows=125053   x26   (PP1)
    request=213324 prev_size=148876 uniform_backed_rows=155648
        reserved_backing_rows=213324 store_bound_rows=213325          (PP0)
    request=92162  prev_size=132408 uniform_backed_rows=133120
        reserved_backing_rows=132408 store_bound_rows=132409          (PP2)

``uniform_backed_rows`` is CHUNK-GRANULAR: it is
``min over buffers of (committed_bytes // row_bytes) * tokens_per_row``
(kv_vmm_backing.py:1428-1459), and the arena commits whole chunks. So it
legitimately sits up to one commit chunk per buffer ABOVE the reservation's row
count. ``_physical_backed_rows`` says so and calls the overshoot "bounded and
in the safe direction ... the only error mode that matters here"
(kv_backing_relief.py:1484-1487) -- true for the SHRINK rung it was written
for, and false for everything that later reused the reading as a ceiling.

THE HOLE, and it is one branch. ``runtime_set_backing_tokens`` validates the
requested size against the reservation only INSIDE the two owner calls:

    if n > backed:   owner.finalize(n)   -> _check_final(n)  n <= reserved
    ...
    released = owner.shrink(n) if n < backed else 0          n <= reserved
    self.size = n

When ``n == backed`` and ``n != prev``, NEITHER call runs and ``self.size = n``
is assigned unvalidated -- and the method's own docstring records the design as
"``self.size`` is assigned either way" (memory_pool.py:3002-3006). Because
``backed`` is the chunk-granular reading, that single unvalidated value is
exactly the one that can sit above the reservation. One such call latches the
pool permanently:

  * ``size`` is now 126976 while the arena can never commit past 125052;
  * ``exposed_rows`` -> ``_reservation_rows`` reads the id-space pool's ``size``
    (kv_backing_relief.py:2970-2994), so the allocator now hands out row ids
    up to 126976 -- above the ceiling AND above ``store_bound_rows`` = 125053,
    which is the bound a CUDA graph baked at capture (#352);
  * the live set fills them, so ``max_live_row`` reaches 126976 and the seam's
    need becomes 126977;
  * every grow to 126977 is refused, because 126977 > 125052.

The gap is then UNCLOSEABLE BY ARITHMETIC, on every round, forever. Window 7
armed 117 times, refused 114, and completed ZERO ``tp_to_pp`` flips in 30.6
minutes of load -- a deterministic outcome, not a flaky seam.

WHY THIS IS NOT A RANK-MEMORY-SPLIT PROBLEM. The reservation is
``cuMemAddressReserve`` VIRTUAL address space (kv_vmm_backing.py:501-504);
physical pages arrive later at ``cuMemCreate``. A rank whose size has drifted
above its arena's VA span is not short of VRAM. Spending real VRAM via
``--rank-gpu-memory-mib`` to paper over a virtual shortfall would also be a
hand-pinned VRAM number, which this project's planner law refuses outright.

WHAT THIS FILE PINS. ``size`` may never be assigned above the arena's
reservation, on ANY branch. The dial keeps its two-axis design -- backing
converges against ``uniform_backed_rows``, ``size`` is assigned either way --
but the assignment is bounded by the ceiling the owner would have enforced had
it been consulted.
"""

import unittest

#: Window 7, PP1, to the row.
W7_RESERVED = 125052
W7_BACKED = 126976  # chunk-granular, 1924 rows above the reservation
W7_PREV = 124704
W7_STORE_BOUND = W7_RESERVED + 1  # 125053, the bound a CUDA graph baked


class _Owner:
    """The arena owner, enforcing exactly what ``_check_final`` enforces."""

    def __init__(self, reserved, page_size=1):
        self.reserved = int(reserved)
        self.page_size = int(page_size)
        self.finalized = []
        self.shrunk = []

    def _check_final(self, final):
        if not (self.page_size <= final <= self.reserved):
            raise ValueError(
                f"final_num_tokens={final} must satisfy "
                f"{self.page_size} <= final <= reserved={self.reserved}"
            )

    def finalize(self, n):
        self._check_final(int(n))
        self.finalized.append(int(n))

    def shrink(self, n, indices=None):
        self._check_final(int(n))
        self.shrunk.append(int(n))
        return 0


class _Pool:
    """Enough of MHATokenToKVPool to drive the real dial method.

    The real ``runtime_set_backing_tokens`` is invoked unbound against this, so
    the BRANCH LOGIC under test is production code, not a restatement of it.
    """

    def __init__(self, size, backed, reserved, page_size=1):
        self.size = int(size)
        self.page_size = int(page_size)
        self._backed = int(backed)
        self._post_capture_owner = _Owner(reserved, page_size)
        self.k_buffer = []
        self.v_buffer = []
        self._kv_buffer_descs = []

    @property
    def uniform_backed_rows(self):
        return self._backed

    @property
    def reserved_backing_rows(self):
        return int(self._post_capture_owner.reserved)

    @property
    def store_bound_rows(self):
        return int(self._post_capture_owner.reserved) + self.page_size


def _dial(pool, n):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool.runtime_set_backing_tokens(pool, n)


class TheUnvalidatedBranch(unittest.TestCase):
    """RED before the fix: n == backed skips both owner calls."""

    def test_size_is_never_assigned_above_the_reservation(self):
        pool = _Pool(W7_PREV, W7_BACKED, W7_RESERVED)
        try:
            _dial(pool, W7_BACKED)
        except ValueError:
            pass  # refusing is a legitimate fix shape
        self.assertLessEqual(
            pool.size,
            pool.reserved_backing_rows,
            f"size={pool.size} was assigned above the arena reservation "
            f"{pool.reserved_backing_rows}; this is the window-7 latch",
        )

    def test_size_never_exceeds_the_graph_baked_store_bound(self):
        """#352: a row id above the captured bound is a device assert."""
        pool = _Pool(W7_PREV, W7_BACKED, W7_RESERVED)
        try:
            _dial(pool, W7_BACKED)
        except ValueError:
            pass
        self.assertLessEqual(pool.size, pool.store_bound_rows)
        self.assertEqual(W7_STORE_BOUND, 125053)


class TheHealthyBranchesAreUntouched(unittest.TestCase):
    """Mutant guard: the fix must not refuse work that was always legal."""

    def test_a_grow_within_the_reservation_still_finalizes(self):
        pool = _Pool(1000, 2000, 8192)
        _dial(pool, 4096)
        self.assertEqual(pool._post_capture_owner.finalized, [4096])
        self.assertEqual(pool.size, 4096)

    def test_a_shrink_within_the_reservation_still_decommits(self):
        pool = _Pool(4096, 4096, 8192)
        _dial(pool, 2048)
        self.assertEqual(pool._post_capture_owner.shrunk, [2048])
        self.assertEqual(pool.size, 2048)

    def test_n_equal_backed_below_the_reservation_is_allowed(self):
        """The ONLY value the fix may newly refuse is one above the ceiling."""
        pool = _Pool(1000, 4096, 8192)
        _dial(pool, 4096)
        self.assertEqual(pool.size, 4096)

    def test_the_noop_branch_still_returns_zero(self):
        pool = _Pool(4096, 4096, 8192)
        self.assertEqual(_dial(pool, 4096), 0)


class TheClampTouchesOnlyTheUnvalidatedBranch(unittest.TestCase):
    """The fix may not swallow a refusal that the owner already owed.

    Both owner calls check the reservation themselves, so anything above the
    ceiling ALREADY raises on the grow and shrink branches::

        n > backed  -> finalize(n) -> _check_final -> ValueError
        n < backed  -> shrink(n)   -> _check_final -> ValueError

    which leaves ``n == backed`` as the only value the clamp can reach. Pinning
    that keeps the blast radius provable: a mutant that clamped at the top of
    the function instead would silently convert both of these loud refusals
    into quiet successes, and these two tests kill it.
    """

    def test_a_grow_above_the_ceiling_still_raises(self):
        pool = _Pool(W7_PREV, W7_BACKED, W7_RESERVED)
        with self.assertRaises(ValueError):
            _dial(pool, W7_BACKED + 1)  # 126977, the window-7 request

    def test_a_shrink_target_above_the_ceiling_still_raises(self):
        pool = _Pool(W7_PREV, W7_BACKED, W7_RESERVED)
        with self.assertRaises(ValueError):
            _dial(pool, W7_RESERVED + 1)  # below backed, above the reservation


class ThePoolWithoutAnArenaIsUnchanged(unittest.TestCase):
    """A pool exposing no reservation keeps exactly its previous behaviour."""

    def test_zero_reservation_does_not_clamp_to_zero(self):
        class _NoArenaPool(_Pool):
            @property
            def reserved_backing_rows(self):
                return 0

        pool = _NoArenaPool(1000, 4096, 8192)
        _dial(pool, 4096)
        self.assertEqual(pool.size, 4096)


if __name__ == "__main__":
    unittest.main()
