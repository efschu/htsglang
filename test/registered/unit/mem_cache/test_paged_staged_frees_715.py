"""#715: the paged alloc paths must spend staged frees before they raise.

THE CRASH THIS IS ABOUT. 2026-08-17 02:18, 512 tokens refused while the
tree reported 147,456 tokens evictable. Same shape as the 2026-08-16
01:46:10 and 13:58:37 events already recorded in ``common.py``: the
eviction reports full delivery, the allocation fails anyway, and the error
message names plenty of memory.

THE MECHANISM, which is not a new one. ``free_group_begin`` is called from
the event loop (``batch_result_processor.py:92`` and ``:741``). While that
window is open, ``PagedTokenToKVPoolAllocator.free`` appends to
``free_group`` instead of extending ``free_pages`` (``allocator/paged.py:
293-308``), so the pages are in neither ``free_pages`` nor
``release_pages`` and ``available_size`` cannot see them -- while the tree
has already counted them as evicted. That is #681's THIRD ROOT, and
``BaseTokenToKVPoolAllocator.flush_free_group`` (``allocator/base.py:208``)
is its remedy.

WHY IT STILL CRASHED. The remedy was wired into ``alloc_token_slots``
only. Every paged path -- ``alloc_paged_token_slots_extend``,
``alloc_paged_token_slots_decode`` -- reached its raise without ever asking
whether the pages it needed were sitting staged. The relief net WAS
extended to those paths under "#681 RULE 3: every alloc path reachable from
prefill admission gets the same net"; the third root was not carried across
with it. So this is one root, wired on one of the paths that needs it.

These pins are the asymmetry, stated as behaviour: with pages staged and
nothing else available, the paged extend path must allocate rather than
raise.

ON THE HYBRID-SUB-POOL CANDIDATE. The candidate under test was that
``_evict_leaf_node``'s ``allocator.free(x.value)`` might route rows to one
sub-pool of a ``HybridLinearKVPool`` while ``available_size``/``alloc``
read another. ``TestAccountingLivesInTheAllocator`` records why that is not
where this divergence comes from: the free/available accounting is entirely
allocator-side, over index bookkeeping that no sub-pool split touches.
"""

import unittest

import torch

from sglang.srt.mem_cache import common as mem_common
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

PAGE = 8
SIZE = 64


class _PagedAllocator(PagedTokenToKVPoolAllocator):
    """The real paged allocator with one pure-Python allocation override.

    Everything this test turns on -- ``free``'s staging branch,
    ``free_group_begin/end``, ``flush_free_group``, ``available_size`` -- is
    the REAL implementation, inherited unmodified. Only ``alloc_extend`` is
    replaced, because the production one dispatches to a CUDA kernel and
    these pins must run with no accelerator. Overriding the allocation while
    inheriting the accounting keeps the stub off the load-bearing path
    (#624).
    """

    def _take(self, need):
        pages_needed = (need + PAGE - 1) // PAGE
        if len(self.free_pages) < pages_needed:
            return None
        take, self.free_pages = (
            self.free_pages[:pages_needed],
            self.free_pages[pages_needed:],
        )
        return take * PAGE

    def alloc_extend(self, *args, **kwargs):
        need = int(args[5]) if len(args) > 5 else int(kwargs["extend_num_tokens"])
        return self._take(need)

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc, **kwargs):
        return self._take(int(seq_lens_cpu.sum()))


class _TreeCache:
    """A tree whose eviction frees INSIDE an open free group.

    That is the production sequence, not a contrivance: the event loop opens
    the group and the eviction runs inside it.
    """

    def __init__(self, allocator, evictable_tokens):
        self.token_to_kv_pool_allocator = allocator
        self._evictable = evictable_tokens
        self.evict_calls = 0

    def is_chunk_cache(self):
        return False

    def evictable_size_(self):
        return len(self._evictable)

    def evict(self, params):
        """Free every evictable token and report full delivery -- truthfully.

        The tokens ARE freed. ``free`` stages them because the group is open,
        which is precisely why a receipt-checking caller cannot catch this:
        the receipt is honest.
        """
        self.evict_calls += 1
        if not self._evictable:
            return _EvictResult(0)
        freed, self._evictable = self._evictable, []
        self.token_to_kv_pool_allocator.free(torch.tensor(freed, dtype=torch.int64))
        return _EvictResult(len(freed))

    def available_and_evictable_str(self):
        return (
            f"Available token size: "
            f"{self.token_to_kv_pool_allocator.available_size()}\n"
            f"Evictable token size: {self.evictable_size_()}\n"
        )

    def pretty_print(self):
        return ""


class _EvictResult:
    def __init__(self, n):
        self.num_tokens_evicted = n
        self.mamba_num_evicted = 0


def _drained_allocator():
    """A paged allocator with every page handed out."""
    allocator = _PagedAllocator(
        size=SIZE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    allocator.free_pages = allocator.free_pages[:0]
    return allocator


def _extend_call(tree_cache, num_tokens):
    ones = torch.ones(1, dtype=torch.int64)
    return mem_common.alloc_paged_token_slots_extend(
        tree_cache,
        prefix_lens=torch.zeros(1, dtype=torch.int64),
        prefix_lens_cpu=torch.zeros(1, dtype=torch.int64),
        seq_lens=ones * num_tokens,
        seq_lens_cpu=ones * num_tokens,
        last_loc=ones * -1,
        extend_num_tokens=num_tokens,
    )


class TestTheStagingDivergenceIsReal(unittest.TestCase):
    """Establish the mechanism on the real allocator before pinning the fix.

    Without this, a green fix-pin could pass for the wrong reason -- e.g. if
    ``free`` never staged at all on this shape.
    """

    def test_a_free_inside_an_open_group_is_invisible_to_available_size(self):
        allocator = _drained_allocator()
        allocator.free_group_begin()
        allocator.free(torch.arange(0, 16, dtype=torch.int64))

        self.assertEqual(
            allocator.available_size(),
            0,
            "the pages must be invisible while staged, or there is no bug to fix",
        )
        self.assertTrue(allocator.free_group, "the free should have been staged")

    def test_flushing_makes_exactly_those_pages_visible(self):
        allocator = _drained_allocator()
        allocator.free_group_begin()
        allocator.free(torch.arange(0, 16, dtype=torch.int64))

        applied = allocator.flush_free_group()

        self.assertEqual(applied, 16)
        self.assertEqual(allocator.available_size(), 16)


class TestThePagedExtendPathSpendsThem(unittest.TestCase):
    """The #715 pin: staged pages must be spent before the raise."""

    def test_extend_allocates_instead_of_raising_when_pages_are_staged(self):
        allocator = _drained_allocator()
        allocator.free_group_begin()
        tree_cache = _TreeCache(allocator, list(range(16)))

        out = _extend_call(tree_cache, 16)

        self.assertIsNotNone(
            out,
            "the paged extend path raised while the tokens it needed were "
            "sitting in the allocator's staged free group",
        )
        self.assertEqual(tree_cache.evict_calls, 1)

    def test_the_error_still_raises_when_nothing_is_staged(self):
        """The fix must not convert a genuine OOM into a silent stall.

        Fail-loud keeps the last word: with no staged pages and nothing
        evictable there is nothing to spend, and the raise must survive.
        """
        allocator = _drained_allocator()
        allocator.free_group_begin()
        tree_cache = _TreeCache(allocator, [])

        with self.assertRaises(RuntimeError):
            _extend_call(tree_cache, 16)

    def test_a_healthy_allocation_does_not_flush(self):
        """The flush is a cold path: it may not disturb a succeeding alloc.

        Pages stay staged here because the first attempt succeeds, which is
        what makes this a check on ordering rather than on the flush itself.
        """
        allocator = _drained_allocator()
        allocator.free_pages = torch.arange(0, 4, dtype=torch.int64)
        allocator.free_group_begin()
        allocator.free(torch.arange(32, 48, dtype=torch.int64))
        staged_before = len(allocator.free_group)
        tree_cache = _TreeCache(allocator, [])

        out = _extend_call(tree_cache, 16)

        self.assertIsNotNone(out)
        self.assertEqual(
            len(allocator.free_group),
            staged_before,
            "a successful allocation must not have touched the staged frees",
        )


class TestThePagedDecodePathSpendsThemToo(unittest.TestCase):
    """The decode twin. The free-group window is opened by the event loop and
    does not care which allocation runs inside it, so the same root reaches
    this path -- which had no net of any kind, not even the relief one."""

    def test_decode_allocates_instead_of_raising_when_pages_are_staged(self):
        allocator = _drained_allocator()
        allocator.free_group_begin()
        tree_cache = _TreeCache(allocator, list(range(16)))

        out = mem_common.alloc_paged_token_slots_decode(
            tree_cache,
            seq_lens=torch.ones(1, dtype=torch.int64) * 16,
            seq_lens_cpu=torch.ones(1, dtype=torch.int64) * 16,
            last_loc=torch.ones(1, dtype=torch.int64) * -1,
        )

        self.assertIsNotNone(out)

    def test_decode_still_raises_when_nothing_is_staged(self):
        allocator = _drained_allocator()
        allocator.free_group_begin()
        tree_cache = _TreeCache(allocator, [])

        with self.assertRaises(RuntimeError):
            mem_common.alloc_paged_token_slots_decode(
                tree_cache,
                seq_lens=torch.ones(1, dtype=torch.int64) * 16,
                seq_lens_cpu=torch.ones(1, dtype=torch.int64) * 16,
                last_loc=torch.ones(1, dtype=torch.int64) * -1,
            )


class TestAccountingLivesInTheAllocator(unittest.TestCase):
    """Why the hybrid-sub-pool candidate is not the source of this divergence.

    ``HybridLinearKVPool`` splits the DATA tensors across a full_kv_pool and
    per-component pools. The free/available accounting these pins exercise is
    index bookkeeping held on the ALLOCATOR (``free_pages`` /
    ``release_pages`` / ``free_group``), and ``available_size`` is computed
    from those two lists alone (``allocator/base.py:187-188``). A free and the
    ``available_size`` that follows it therefore read the same structure no
    matter how the pool splits its tensors underneath.

    This does not say the hybrid pool is free of defects -- only that it
    cannot produce THIS divergence, and that the observed one is fully
    accounted for by the staging window.
    """

    def test_available_size_is_computed_only_from_the_allocator_lists(self):
        import inspect

        from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

        source = inspect.getsource(BaseTokenToKVPoolAllocator.available_size)
        self.assertIn("free_pages", source)
        self.assertIn("release_pages", source)
        self.assertNotIn(
            "_kvcache",
            source,
            "available_size consulting the pool would reopen the sub-pool "
            "routing question this test class closes",
        )

    def test_free_and_available_size_agree_with_no_pool_at_all(self):
        """kvcache=None throughout: the accounting never consults the pool."""
        allocator = _drained_allocator()
        allocator.free(torch.arange(0, 24, dtype=torch.int64))

        self.assertEqual(allocator.available_size(), 24)


if __name__ == "__main__":
    unittest.main()
