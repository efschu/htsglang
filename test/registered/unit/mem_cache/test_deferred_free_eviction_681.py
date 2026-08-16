"""#681 third root: eviction counts tokens the allocator has only STAGED.

THE SPECIMEN (2026-08-16 13:58:37, #693, all three ranks identically)::

    RuntimeError: Out of memory. Try to allocate 512 tokens.
    Available full tokens: 167743 (full_available_size=392 + full_evictable_size=167351)

and **no** ``EVICTION UNDER-DELIVERED`` line. That absence is the evidence, not
a gap in it: ``_eviction_shortfall_note`` returns ``""`` only when
``evicted >= asked``, so eviction reported delivering its full 512 while the
allocator's free count stayed at 392.

THE CHAIN, every hop verified in source:

1. ``_evict_leaf_node`` (mamba_radix_cache.py:915-916) calls
   ``token_to_kv_pool_allocator.free(x.value)`` and counts ``len(x.value)`` as
   evicted. Same for the tombstone route, ``_free_tombstone_leaf`` (:1705-1706).
2. ``TokenToKVPoolAllocator.free`` (allocator/token.py:67-80) applies the pages
   only ``if self.is_not_in_free_group``; otherwise it appends to
   ``self.free_group``.
3. ``available_size`` (allocator/token.py:52-54) is
   ``len(free_pages) + len(release_pages)`` -- the staged list is in NEITHER,
   and ``alloc`` (:56-61) compares against ``len(free_pages)`` alone.
4. ``free_group_begin`` is called from inside the event loop
   (batch_result_processor.py:92 and :741), so an eviction landing in that
   window frees into the staging list.

So the receipt is true about the TREE and false about the POOL: the nodes are
gone and their tokens counted, and the pages are sitting in
``allocator.free_group`` waiting for ``free_group_end()``.

WHY IT IS NOT A POOL-TIER MISMATCH. ``release_pages`` IS counted by
``available_size``, so a tier split would still show the tokens. And it is not
a race: the group opens at a fixed point in a replicated event loop, which is
why all three ranks printed byte-identical numbers.

THE FIX, and why this one. ``free_group_end`` is *pure batching* --
``self.free(torch.cat(self.free_group))``, one concat and one ``_notify_free``
(allocator/base.py:203-206). It defers nothing for safety: no in-flight
reference, no graph-replay barrier, and the same iteration applies the pages a
few lines later regardless. So the staged pages are payable NOW, and the honest
repair is to make them payable rather than to teach the counter to lie less:
``flush_free_group()`` applies them WITHOUT closing the group, so the caller's
batching window and its later ``free_group_end()`` are untouched.
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


class FreeGroupAllocator:
    """The real ``TokenToKVPoolAllocator`` free/alloc semantics, minimally.

    Mirrors allocator/token.py: ``free`` stages while a group is open,
    ``available_size`` counts only what has been applied, and ``alloc`` draws
    from the applied pages. Tensor-based, because the PRODUCTION
    ``flush_free_group`` is bound onto this stand-in rather than reimplemented
    -- the method under test has to be the real one.
    """

    flush_free_group = BaseTokenToKVPoolAllocator.flush_free_group

    def __init__(self, free_pages: int):
        self.free_pages = torch.arange(free_pages, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)
        self.free_group = []
        self.is_not_in_free_group = True
        self.alloc_calls = []

    # -- the two the specimen turns on ---------------------------------
    def free(self, index: torch.Tensor):
        if index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.free_pages = torch.cat((self.free_pages, index))
        else:
            self.free_group.append(index)

    def available_size(self):
        return int(self.free_pages.numel() + self.release_pages.numel())

    def alloc(self, n):
        self.alloc_calls.append(n)
        if n > int(self.free_pages.numel()):
            return None
        out, self.free_pages = self.free_pages[:n], self.free_pages[n:]
        return out

    # -- the group protocol --------------------------------------------
    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            staged, self.free_group = self.free_group, []
            self.free(torch.cat(staged))


class _Evicted:
    def __init__(self, n):
        self.num_tokens_evicted = n


class StagingTreeCache:
    """A tree whose eviction frees into the allocator and counts what it freed.

    That is exactly what ``_evict_leaf_node`` does; with a group open the frees
    land in the staging list and the count becomes a promise the pool cannot
    pay.
    """

    def __init__(self, allocator, evictable: int):
        self.token_to_kv_pool_allocator = allocator
        self._evictable = evictable
        self.uniform_avail_floor = None
        self.pretty_printed = 0

    def evictable_size(self):
        return self._evictable

    def is_chunk_cache(self):
        return False

    def is_tree_cache(self):
        return True

    def evict(self, params):
        asked = int(params.num_tokens)
        pages = torch.arange(10_000, 10_000 + asked, dtype=torch.int64)
        self.token_to_kv_pool_allocator.free(pages)
        self._evictable -= asked
        return _Evicted(asked)

    def pretty_print(self):
        self.pretty_printed += 1
        return ""

    def available_and_evictable_str(self):
        a = self.token_to_kv_pool_allocator.available_size()
        return (
            f"Available full tokens: {a + self._evictable} "
            f"(full_available_size={a} + full_evictable_size={self._evictable})"
        )


def _specimen(free_pages=392, evictable=167351):
    alloc = FreeGroupAllocator(free_pages=free_pages)
    tree = StagingTreeCache(alloc, evictable=evictable)
    return alloc, tree


class TheStagedFreeIsPayable(unittest.TestCase):
    """The root: tokens counted as evicted must be reachable by the alloc that
    the count was produced for."""

    def test_the_allocation_succeeds_once_the_staged_frees_are_applied(self):
        alloc, tree = _specimen()
        alloc.free_group_begin()  # the event-loop window
        got = alloc_token_slots(tree, 512)
        self.assertIsNotNone(
            got,
            "the allocation still fails while 512 evicted tokens sit in "
            "allocator.free_group -- the counted-but-unpayable state",
        )
        self.assertEqual(len(got), 512)

    def test_the_group_stays_open_across_the_flush(self):
        """The caller opened it and still owns it: flushing must not close a
        batching window someone else is inside."""
        alloc, tree = _specimen()
        alloc.free_group_begin()
        alloc_token_slots(tree, 512)
        self.assertFalse(
            alloc.is_not_in_free_group,
            "the flush closed the caller's free group; its later frees would "
            "stop batching and its free_group_end would be a no-op it did not "
            "ask for",
        )

    def test_the_callers_group_end_is_still_safe_afterwards(self):
        """No double-free: what was flushed must not be applied twice."""
        alloc, tree = _specimen()
        alloc.free_group_begin()
        alloc_token_slots(tree, 512)
        before = alloc.available_size()
        alloc.free_group_end()
        self.assertEqual(alloc.available_size(), before)


class TheHealthyPathIsUntouched(unittest.TestCase):
    """No group open, or nothing staged: byte-identical behaviour."""

    def test_no_group_open_allocates_as_before(self):
        alloc, tree = _specimen(free_pages=1024)
        got = alloc_token_slots(tree, 512)
        self.assertEqual(len(got), 512)
        self.assertEqual(alloc.alloc_calls, [512])

    def test_a_pool_that_is_genuinely_empty_still_raises(self):
        """Fail-loud is preserved: nothing staged, nothing evictable."""
        alloc = FreeGroupAllocator(free_pages=0)
        tree = StagingTreeCache(alloc, evictable=0)

        def _nothing(params):
            return _Evicted(0)

        tree.evict = _nothing
        with self.assertRaises(RuntimeError):
            alloc_token_slots(tree, 512)

    def test_an_open_group_with_nothing_staged_still_raises(self):
        alloc = FreeGroupAllocator(free_pages=0)
        tree = StagingTreeCache(alloc, evictable=0)

        def _nothing(params):
            return _Evicted(0)

        tree.evict = _nothing
        alloc.free_group_begin()
        with self.assertRaises(RuntimeError):
            alloc_token_slots(tree, 512)


class TheRealAllocatorCarriesTheFlush(unittest.TestCase):
    """The stand-in above is only honest if the production class has the same
    capability, on the base so every subclass inherits it."""

    def test_the_base_allocator_defines_flush_free_group(self):
        from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

        self.assertTrue(hasattr(BaseTokenToKVPoolAllocator, "flush_free_group"))

    def test_the_flush_applies_and_keeps_the_group_open(self):
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        a = TokenToKVPoolAllocator.__new__(TokenToKVPoolAllocator)
        a.free_pages = torch.arange(4, dtype=torch.int64)
        a.release_pages = torch.empty(0, dtype=torch.int64)
        a.free_group = []
        a.is_not_in_free_group = True
        a.need_sort = False
        a._free_listeners = []
        a.free_group_begin()
        a.free(torch.arange(100, 108, dtype=torch.int64))
        self.assertEqual(int(a.available_size()), 4, "staged frees must not count")
        applied = a.flush_free_group()
        self.assertEqual(applied, 8)
        self.assertEqual(int(a.available_size()), 12)
        self.assertFalse(a.is_not_in_free_group)
        self.assertEqual(a.free_group, [])


if __name__ == "__main__":
    unittest.main()
