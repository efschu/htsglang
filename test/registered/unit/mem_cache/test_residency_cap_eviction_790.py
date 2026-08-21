"""#790: eviction counts slots a residency cap takes straight back.

THE SPECIMEN (2026-08-21 01:54:55, PP0, 4m55s after a ``tp_to_pp`` flip)::

    [01:54:51] PHASE-FLIP DONE tp_to_pp (epoch 22) in 3964.6 ms: 160822 live slots
    [01:54:55] extend allocation failed and NO relief provider is registered
    RuntimeError: Out of memory. Try to allocate 512 tokens.
    Available full tokens: 138089 (full_available_size=189 + full_evictable_size_=137900)
    Available mamba: 20 (available_size=3 + component_evictable_size_=17)

and **no** ``EVICTION UNDER-DELIVERED`` line -- the same absence that was the
evidence in #681, and here it is the evidence again:
``_eviction_shortfall_note`` returns ``""`` only when ``evicted >= asked``, so
eviction reported delivering its full 512 while ``full_available_size`` stayed
at 189. The admission gate one iteration earlier read the same pool as healthy::

    [01:54:55] #788 PP-ADMISSION verdict=ADMIT ... avail=189 evictable=138412

138412 -> 137900 across the raise is a drop of exactly 512: the eviction ran,
the tree's books moved, and the allocator's did not.

THE CHAIN, every hop verified in source:

1. ``FullComponent.evict_component``
   (mem_cache/unified_cache_components/full_component.py:115-119) calls
   ``self._free_full(cd.value)``, sets ``freed = len(cd.value)`` and decrements
   ``component_evictable_size_`` by it. The count is taken the instant the free
   is HANDED OVER, not when the pool receives it.
2. ``UnifiedRadixCache.evict`` (mem_cache/unified_radix_cache.py:845) returns
   that tracker as ``num_tokens_evicted``; ``evict_from_tree_cache``
   (mem_cache/common.py) returns it unchanged; ``alloc_token_slots`` fed it to
   ``_eviction_shortfall_note`` as the delivery.
3. ``KvRowCap.engage`` (managers/kv_backing_relief.py:379) subscribes to the
   allocator's free listener, and ``KvRowCap._apply`` (:490-512) moves every id
   above the cap out of ``free_pages``/``release_pages`` into ``_withheld`` --
   ON EVERY FREE, which is the point of the hook: an id above the cap that
   re-entered the free list is an id the next allocation hands to a kernel
   writing into unmapped memory.
4. ``TokenToKVPoolAllocator.available_size`` (allocator/token.py:52-54) counts
   neither list's confiscated entries, because they are in neither list.

So the receipt is true about the TREE and false about the POOL, exactly as in
#681's third root -- but the confiscator is different and the #681 repair does
not reach it: ``flush_free_group`` has nothing staged (the boot log carries no
"#681 third root" line), and the withheld ids are not payable at all.

WHY THE BOOT REACHED THAT STATE. The rung engaged its cap at 01:53:40 --
"KV-BACKING released 280 MiB by backing 137135 rows instead of 161792 ... 24243
ids withheld from the allocator" -- and the flip at 01:54:51 then carried
160822 live slots back across the FULL 161792-id space. From that point most of
the tree's evictable tokens sat above row 137135, and peeling them freed the
tree while paying the pool nothing.

THE FIX, and why this one. The cap is not the bug: refusing to hand out an id
whose page is unmapped is the only safe thing it can do. The bug is that the
peel BELIEVED A RECEIPT IT DID NOT MEASURE. ``alloc_token_slots`` now measures
delivery as the growth of :func:`payable_size` across the eviction -- available
plus what an open free group still owes, so #681's staging keeps reading as the
delivery it is -- and when the pool received less than it asked for, it peels
again. That is enough on its own: the confiscated ids are a REGION, and on this
specimen ~114k evictable tokens sat BELOW the cap and were payable the whole
time.
"""

import unittest

import torch

from sglang.srt.managers.kv_backing_relief import KvRowCap
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

#: The specimen's numbers, scaled to something a CPU test can hold. The shape
#: is what matters and it is preserved exactly: a pool whose live set spans the
#: WHOLE id space, a cap partway up it, and a tree whose oldest leaves -- the
#: ones the peel reaches first -- are the ones above the cap.
POOL = 4096
CAP = 3072
CHUNK = 512


def _cpu_allocator(size: int = POOL) -> TokenToKVPoolAllocator:
    """The production allocator on the CPU, with no KV tensors behind it.

    ``kvcache`` is only ever dereferenced by ``get_cpu_copy``/``load_cpu_copy``,
    which nothing on the allocation path calls; ``free``/``alloc``/
    ``available_size`` and the free-listener protocol are all the real ones.
    """
    return TokenToKVPoolAllocator(
        size=size, dtype=torch.int64, device="cpu", kvcache=None, need_sort=False
    )


class ConfiscatedTreeCache:
    """A tree that frees its leaves into the allocator and counts what it freed.

    That IS ``FullComponent.evict_component``: free the leaf's value, add
    ``len(value)`` to the tracker, subtract it from ``component_evictable_size_``.
    Nothing here is more careful than production -- in particular this stand-in
    does NOT check whether the pool received the slots, because the class it
    stands in for does not check either. That omission is the bug.

    The leaves are handed out oldest-first from ``high_ids`` (slots above the
    cap, which the flip's carry left resident) and only then from ``low_ids``,
    which is the eviction order the specimen's LRU frontier had.
    """

    def __init__(self, allocator, high_ids, low_ids):
        self.token_to_kv_pool_allocator = allocator
        self._leaves = [torch.tensor(ids, dtype=torch.int64) for ids in high_ids]
        self._leaves += [torch.tensor(ids, dtype=torch.int64) for ids in low_ids]
        self._evictable = sum(int(t.numel()) for t in self._leaves)
        self.uniform_avail_floor = None
        self.evict_calls = []
        self.pretty_printed = 0

    # -- BasePrefixCache surface the alloc path touches ----------------
    def evictable_size(self):
        return self._evictable

    def full_evictable_size(self):
        return self._evictable

    def is_chunk_cache(self):
        return False

    def is_tree_cache(self):
        return True

    def evict(self, params):
        asked = int(params.num_tokens)
        self.evict_calls.append(asked)
        counted = 0
        while counted < asked and self._leaves:
            leaf = self._leaves.pop(0)
            self.token_to_kv_pool_allocator.free(leaf)
            n = int(leaf.numel())
            counted += n
            self._evictable -= n
        return type("R", (), {"num_tokens_evicted": counted})()

    def pretty_print(self):
        self.pretty_printed += 1
        return ""

    def available_and_evictable_str(self):
        a = int(self.token_to_kv_pool_allocator.available_size())
        return (
            f"Available full tokens: {a + self._evictable} "
            f"(full_available_size={a} + full_evictable_size_={self._evictable})\n"
        )


def _specimen(free_below_cap: int = 0):
    """A capped pool whose evictable frontier starts above the cap.

    Returns ``(allocator, cap, tree)``. The allocator is drained to
    ``free_below_cap`` payable slots first, so the eviction is the only source
    the allocation has -- which is the specimen's state (189 free against a 512
    ask).
    """
    allocator = _cpu_allocator()
    cap = KvRowCap(allocator)
    cap.engage(CAP)
    # Everything above the cap is withheld now; drain the rest down to the
    # residue the specimen had.
    payable = int(allocator.available_size())
    if payable > free_below_cap:
        allocator.alloc(payable - free_below_cap)

    high = [
        list(range(CAP + 1 + i * CHUNK, CAP + 1 + (i + 1) * CHUNK)) for i in range(2)
    ]
    low = [list(range(1 + i * CHUNK, 1 + (i + 1) * CHUNK)) for i in range(4)]
    return allocator, cap, ConfiscatedTreeCache(allocator, high, low)


class TheCapConfiscatesWhatTheTreeCounted(unittest.TestCase):
    """The mechanism itself, against the production cap and allocator.

    If this class ever goes green without the rest, the specimen's premise is
    gone and the fix below is guarding nothing.
    """

    def test_a_free_above_the_cap_moves_no_availability(self):
        allocator = _cpu_allocator()
        cap = KvRowCap(allocator)
        cap.engage(CAP)
        allocator.alloc(int(allocator.available_size()))
        self.assertEqual(int(allocator.available_size()), 0)

        allocator.free(torch.arange(CAP + 1, CAP + 1 + CHUNK, dtype=torch.int64))
        self.assertEqual(
            int(allocator.available_size()),
            0,
            "the cap's free listener must take the freed ids straight back; "
            "without that this whole ticket has no mechanism",
        )
        self.assertGreaterEqual(cap.withheld, CHUNK)

    def test_a_free_below_the_cap_does_move_availability(self):
        """The counterpart, so the test above cannot pass by the pool being
        broken in some other way."""
        allocator = _cpu_allocator()
        cap = KvRowCap(allocator)
        cap.engage(CAP)
        allocator.alloc(int(allocator.available_size()))
        allocator.free(torch.arange(1, 1 + CHUNK, dtype=torch.int64))
        self.assertEqual(int(allocator.available_size()), CHUNK)


class TheAllocationSurvivesAConfiscatedPeel(unittest.TestCase):
    """The root: an eviction the pool did not receive is not a delivery, and
    the peel must keep going until it is."""

    def test_the_allocation_succeeds_by_peeling_past_the_capped_region(self):
        allocator, cap, tree = _specimen()
        got = alloc_token_slots(tree, CHUNK)
        self.assertIsNotNone(
            got,
            "the allocation still fails while the tree reports it freed 512 "
            "tokens the residency cap confiscated -- the counted-but-"
            "unreceived state",
        )
        self.assertEqual(len(got), CHUNK)
        self.assertTrue(
            all(int(i) <= CAP for i in got),
            "every slot handed out must be below the cap; an id above it names "
            "a page the rung has unmapped",
        )

    def test_the_peel_stops_as_soon_as_the_pool_has_been_paid(self):
        """Cache is not free: the rung may spend only what the shortfall needs."""
        allocator, cap, tree = _specimen()
        alloc_token_slots(tree, CHUNK)
        self.assertLessEqual(
            len(tree.evict_calls),
            4,
            f"the peel took {tree.evict_calls} rounds to cover a 2-leaf "
            f"confiscated region; it is walking the tree rather than escalating",
        )
        self.assertGreaterEqual(
            tree.evictable_size(),
            CHUNK,
            "the rung evicted the whole tree to fund one chunk",
        )

    def test_the_error_names_the_confiscator_when_nothing_is_payable(self):
        """Fail-loud is preserved, and now it is a diagnosis.

        Every leaf sits above the cap, so no peel can ever pay. The raise must
        still happen -- and it must say why, which is the line the specimen did
        not have.
        """
        allocator = _cpu_allocator()
        cap = KvRowCap(allocator)
        cap.engage(CAP)
        allocator.alloc(int(allocator.available_size()))
        high = [
            list(range(CAP + 1 + i * CHUNK, CAP + 1 + (i + 1) * CHUNK))
            for i in range(2)
        ]
        tree = ConfiscatedTreeCache(allocator, high, [])

        with self.assertRaises(RuntimeError) as raised:
            alloc_token_slots(tree, CHUNK)
        msg = str(raised.exception)
        self.assertIn("EVICTION UNDER-DELIVERED", msg)
        self.assertIn("RESIDENCY CAP IS ENGAGED", msg)
        self.assertIn(str(cap.withheld), msg)


class TheHealthyPathIsUntouched(unittest.TestCase):
    """No cap, or a pool that can simply pay: behaviour as before."""

    def test_an_uncapped_pool_allocates_without_peeling_twice(self):
        allocator = _cpu_allocator()
        tree = ConfiscatedTreeCache(
            allocator,
            [],
            [list(range(1 + i * CHUNK, 1 + (i + 1) * CHUNK)) for i in range(4)],
        )
        allocator.alloc(int(allocator.available_size()))
        got = alloc_token_slots(tree, CHUNK)
        self.assertEqual(len(got), CHUNK)
        self.assertEqual(
            tree.evict_calls, [CHUNK], "one eviction paid; there was nothing to retry"
        )

    def test_a_pool_with_room_evicts_nothing(self):
        allocator = _cpu_allocator()
        tree = ConfiscatedTreeCache(allocator, [], [])
        got = alloc_token_slots(tree, CHUNK)
        self.assertEqual(len(got), CHUNK)
        self.assertEqual(tree.evict_calls, [])

    def test_a_genuinely_empty_pool_still_raises(self):
        allocator = _cpu_allocator()
        allocator.alloc(int(allocator.available_size()))
        tree = ConfiscatedTreeCache(allocator, [], [])
        with self.assertRaises(RuntimeError):
            alloc_token_slots(tree, CHUNK)


class TheRungRefusesUnderARankUniformFloor(unittest.TestCase):
    """#616g: under an active floor the radix trees are replicas that must peel
    identically, and this rung's round count is rank-local. It declines."""

    def test_an_active_floor_declines_the_extra_peel(self):
        allocator, cap, tree = _specimen()
        tree.uniform_avail_floor = 0
        tree.uniform_admitted_since_floor = 0
        with self.assertRaises(RuntimeError):
            alloc_token_slots(tree, CHUNK)
        self.assertEqual(
            tree.evict_calls,
            [CHUNK],
            "the rung peeled again under a published floor; that is the "
            "rank-local peel #616g exists to forbid",
        )


if __name__ == "__main__":
    unittest.main()
